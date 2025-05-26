from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import tomllib


CONFIG_DIR = Path(__file__).parent.parent.parent / 'config'


@dataclass
class Config:
    name: str
    path: Path
    data: dict
    scope: tuple[str, ...] = ()

    @classmethod
    def factory(cls, name: str):
        path = (CONFIG_DIR / name).with_suffix('.toml')
        with path.open('rb') as handle:
            data = tomllib.load(handle)
        return cls(name, path, data)

    def __getattr__(self, key: str):
        data = self.data[key]
        if isinstance(data, dict):
            return replace(self, data=data)
        return data
