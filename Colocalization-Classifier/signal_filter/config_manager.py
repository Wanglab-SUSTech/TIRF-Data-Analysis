"""配置管理模块"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """JSON 配置管理器，支持版本控制和配置迁移"""
    
    CONFIG_VERSION = "1.0"
    
    def __init__(self, filename: str = ".signal_filter_config.json"):
        """
        初始化配置管理器
        
        Args:
            filename: 配置文件名
        """
        self.config_path = Path.home() / filename

    def load(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典，加载失败返回空字典
        """
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
                return self._validate_and_migrate(raw_config)
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            return {}
        except OSError as e:
            print(f"读取配置文件失败: {e}")
            return {}
        except Exception as e:
            print(f"加载配置时发生未知错误: {e}")
            return {}

    def save(self, data: Dict[str, Any]) -> bool:
        """
        保存配置文件
        
        Args:
            data: 要保存的配置数据
            
        Returns:
            保存是否成功
        """
        try:
            versioned_data = {
                'version': self.CONFIG_VERSION,
                'timestamp': time.time(),
                'data': data
            }
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(versioned_data, f, ensure_ascii=False, indent=2)
            return True
            
        except OSError as e:
            print(f"配置保存失败: {e}")
            return False
        except Exception as e:
            print(f"保存配置时发生未知错误: {e}")
            return False

    def _validate_and_migrate(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证并迁移配置
        
        Args:
            raw_config: 原始配置
            
        Returns:
            验证后的配置数据
        """
        # 兼容旧版本（没有版本号的配置）
        if 'version' not in raw_config:
            return raw_config
        
        version = raw_config.get('version')
        data = raw_config.get('data', {})
        
        # 版本匹配，直接返回
        if version == self.CONFIG_VERSION:
            return data
        
        # 未来版本迁移逻辑
        # if version == "0.9":
        #     data = self._migrate_from_0_9(data)
        
        return data

    def backup(self) -> bool:
        """
        备份当前配置文件
        
        Returns:
            备份是否成功
        """
        if not self.config_path.exists():
            return False
        
        try:
            backup_path = self.config_path.with_suffix('.json.bak')
            import shutil
            shutil.copy2(self.config_path, backup_path)
            return True
        except Exception as e:
            print(f"备份配置失败: {e}")
            return False
