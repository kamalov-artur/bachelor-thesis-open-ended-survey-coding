import os
import uuid
from dataclasses import dataclass
from pathlib import Path
import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class GigaChatConfig:
    auth_url: str
    chat_url: str
    scope: str
    model: str
    verify_ssl: bool
    connect_timeout: int
    read_timeout: int
    use_env_proxies: bool
    temperature: float
    max_tokens: int

    @property
    def timeout(self):
        return (self.connect_timeout, self.read_timeout)

class GigaChatClient:

    def __init__(self, config, env_path='.env'):
        load_dotenv(env_path)
        self.config = config
        self.credentials = os.getenv('GIGACHAT_CREDENTIALS', '').strip()
        self.session = requests.Session()
        self.session.trust_env = config.use_env_proxies
        self._access_token = None

    def chat(self, messages):
        headers = {'Authorization': f'Bearer {self._get_token()}', 'Content-Type': 'application/json'}
        payload = {'model': self.config.model, 'messages': messages, 'temperature': self.config.temperature, 'max_tokens': self.config.max_tokens}
        response = self.session.post(self.config.chat_url, headers=headers, json=payload, timeout=self.config.timeout, verify=self.config.verify_ssl)
        if response.status_code == 401:
            self._access_token = None
            headers['Authorization'] = f'Bearer {self._get_token()}'
            response = self.session.post(self.config.chat_url, headers=headers, json=payload, timeout=self.config.timeout, verify=self.config.verify_ssl)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _get_token(self):
        if self._access_token:
            return self._access_token
        headers = {'Authorization': f'Basic {self.credentials}', 'RqUID': str(uuid.uuid4()), 'Content-Type': 'application/x-www-form-urlencoded'}
        response = self.session.post(self.config.auth_url, headers=headers, data={'scope': self.config.scope}, timeout=self.config.timeout, verify=self.config.verify_ssl)
        response.raise_for_status()
        self._access_token = response.json()['access_token']
        return self._access_token

def gigachat_config_from_dict(cfg):
    load_dotenv('.env')
    verify_env = os.getenv('GIGACHAT_VERIFY_SSL', '').strip().lower()
    request_timeout = int(cfg['request_timeout'])
    return GigaChatConfig(auth_url=cfg['auth_url'], chat_url=cfg['chat_url'], scope=os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS'), model=os.getenv('GIGACHAT_MODEL', 'GigaChat'), verify_ssl=verify_env not in {'0', 'false', 'no'}, connect_timeout=int(cfg.get('connect_timeout', request_timeout)), read_timeout=int(cfg.get('read_timeout', request_timeout)), use_env_proxies=bool(cfg.get('use_env_proxies', False)), temperature=float(cfg['temperature']), max_tokens=int(cfg['max_tokens']))
