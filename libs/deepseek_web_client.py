# deepseek_web_client.py
import base64
import json
import warnings
from typing import Any, Dict, Optional, List, Generator, Union
from enum import Enum

import requests
from urllib3.exceptions import InsecureRequestWarning

from libs.deepseek_wasm import DeepSeekHashWasm

# Suprimir advertencias de SSL inseguro
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

BASE_URL = "https://chat.deepseek.com"
CREATE_SESSION_URL = f"{BASE_URL}/api/v0/chat_session/create"
CHAT_COMPLETION_URL = f"{BASE_URL}/api/v0/chat/completion"
CREATE_POW_CHALLENGE_URL = f"{BASE_URL}/api/v0/chat/create_pow_challenge"


class ModelType(str, Enum):
    """Modelos disponibles en DeepSeek"""
    DEFAULT = "default"
    EXPERT = "expert"


class Message:
    """Representa un mensaje en la conversación"""
    def __init__(self, id: int, role: str, content: str, parent_id: Optional[int] = None):
        self.id = id
        self.role = role
        self.content = content
        self.parent_id = parent_id


class Chat:
    """
    Objeto que representa una conversación con DeepSeek.
    El modelo se define al crear la conversación y no puede cambiarse.
    """
    
    def __init__(
        self,
        client: "DeepSeekWebClient",
        session_id: str,
        model: ModelType,
    ):
        self.client = client
        self.session_id = session_id
        self.model = model
        self.messages: List[Message] = []
        self.last_message_id: Optional[int] = None

    def send(
        self,
        prompt: str,
        *,
        thinking_enabled: bool = False,
        search_enabled: bool = True,
    ) -> str:
        """
        Envía un mensaje y retorna la respuesta completa.
        """
        response_text, message_id = self.client._chat_completion(
            chat_session_id=self.session_id,
            prompt=prompt,
            parent_message_id=self.last_message_id,
            model_type=self.model.value,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
        )
        
        # Guardar mensaje del usuario
        user_msg_id = len(self.messages) + 1
        self.messages.append(Message(
            id=user_msg_id,
            role="USER",
            content=prompt,
            parent_id=self.last_message_id,
        ))
        
        # Guardar respuesta del asistente
        if message_id:
            self.last_message_id = message_id
            self.messages.append(Message(
                id=message_id,
                role="ASSISTANT",
                content=response_text,
                parent_id=self.last_message_id,
            ))
        
        return response_text[:-8] # -8, for some reason it always returns with "FINISHED" at the end.

    def get_history(self) -> List[Dict[str, str]]:
        """Retorna el historial de la conversación"""
        return [
            {"role": msg.role.lower(), "content": msg.content}
            for msg in self.messages
        ]

    def clear(self) -> None:
        """Limpia el historial de la conversación"""
        self.messages.clear()
        self.last_message_id = None


class DeepSeekWebClient:
    def __init__(
        self,
        bearer_token: str,
        *,
        verify_ssl: bool = False,
        referer: str = "https://chat.deepseek.com",
        wasm_path: str = "sha3_wasm_bg.7b9ca65ddd.wasm",
    ) -> None:
        self.verify_ssl = verify_ssl

        if not verify_ssl:
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        self.session = requests.Session()
        self.session.headers.update({
            "Accept-Language": "es-ES,es;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Origin": "https://chat.deepseek.com",
            "Pragma": "no-cache",
            "Referer": referer,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
            "accept": "*/*",
            "authorization": f"Bearer {bearer_token}",
            "content-type": "application/json",
            "sec-ch-ua": "\"Not:A-Brand\";v=\"99\", \"Google Chrome\";v=\"145\", \"Chromium\";v=\"145\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "x-app-version": "20241129.1",
            "x-client-locale": "es",
            "x-client-platform": "web",
            "x-client-timezone-offset": "7200",
            "x-client-version": "2.0.0",
        })

        self.deepseek_hash = DeepSeekHashWasm(wasm_path)

    def create_chat(self, model: ModelType = ModelType.DEFAULT) -> Chat:
        """Crea una nueva conversación con el modelo especificado"""
        session_id = self._create_chat_session()
        return Chat(self, session_id, model=model)

    def _create_chat_session(self) -> str:
        """Crea una nueva sesión en el servidor"""
        resp = self.session.post(
            CREATE_SESSION_URL,
            json={},
            verify=self.verify_ssl,
        )
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json()
        chat_session = (
            data.get("data", {})
                .get("biz_data", {})
                .get("chat_session", {})
        )
        chat_session_id = chat_session.get("id")
        if not chat_session_id:
            raise RuntimeError(f"No chat_session_id in response: {data}")
        return chat_session_id

    def _create_pow_challenge(self, target_path: str = "/api/v0/chat/completion") -> Dict[str, Any]:
        """Obtiene un desafío PoW del servidor"""
        payload = {"target_path": target_path}
        resp = self.session.post(
            CREATE_POW_CHALLENGE_URL,
            json=payload,
            verify=self.verify_ssl,
        )
        resp.raise_for_status()
        data = resp.json()
        challenge = (
            data.get("data", {})
                .get("biz_data", {})
                .get("challenge", {})
        )
        if not challenge:
            raise RuntimeError(f"No challenge in response: {data}")
        return challenge

    def _solve_pow(self, challenge: Dict[str, Any]) -> str:
        """Resuelve el desafío PoW usando WASM"""
        answer = self.deepseek_hash.calculate_hash(
            algorithm=challenge["algorithm"],
            challenge=challenge["challenge"],
            salt=challenge["salt"],
            difficulty=challenge["difficulty"],
            expire_at=challenge["expire_at"],
        )
        
        if answer is None:
            raise RuntimeError("WASM PoW solver returned no value")

        pow_obj = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }

        raw = json.dumps(pow_obj, separators=(",", ":"))
        return base64.b64encode(raw.encode('utf-8')).decode('ascii')

    def _chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: Optional[int] = None,
        model_type: str = "default",
        thinking_enabled: bool = False,
        search_enabled: bool = True,
    ) -> tuple[str, Optional[int]]:
        """Envía un mensaje y retorna (respuesta, message_id)"""
        challenge = self._create_pow_challenge(target_path="/api/v0/chat/completion")
        x_ds_pow_response = self._solve_pow(challenge)

        headers = {"x-ds-pow-response": x_ds_pow_response}

        payload = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }
        
        resp = self.session.post(
            CHAT_COMPLETION_URL,
            headers=headers,
            json=payload,
            verify=self.verify_ssl,
            stream=True
        )
        resp.raise_for_status()

        full_response = []
        message_id = None
        
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
                
            if line.startswith('data:'):
                data_content = line[5:].strip()
                
                if data_content and data_content != '[DONE]':
                    try:
                        parsed = json.loads(data_content)
                        
                        if 'v' in parsed and isinstance(parsed['v'], dict):
                            if 'response' in parsed['v']:
                                msg_id = parsed['v']['response'].get('message_id')
                                if msg_id:
                                    message_id = msg_id
                        
                        content = None
                        
                        if isinstance(parsed.get('v'), str):
                            content = parsed['v']
                        elif isinstance(parsed.get('v'), dict):
                            fragments = parsed['v'].get('response', {}).get('fragments', [])
                            for fragment in fragments:
                                if fragment.get('type') == 'RESPONSE':
                                    fragment_content = fragment.get('content', '')
                                    if fragment_content:
                                        content = fragment_content
                                        break
                        
                        if content is None and parsed.get('p') == 'response/fragments/-1/content':
                            content = parsed.get('v')
                        
                        if content:
                            full_response.append(content)
                    
                    except json.JSONDecodeError:
                        pass
        
        return ''.join(full_response), message_id