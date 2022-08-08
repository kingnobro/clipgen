from typing import Union
import os
from typing import Union
from pathlib import Path
from ffrecord.torch import Dataset, DataLoader

DATA_DIR = None
DEFAULT_DATA_DIR = Path("./")


def set_data_dir(path: Union[str, os.PathLike]) -> None:
    """
    设置数据集存放的主目录

    我们会优先使用通过 ``set_data_dir`` 设置的路径，如果没有则会去使用环境变量 ``HFAI_DATASETS_DIR`` 的值。
    两者都没有设置的情况下，使用默认目录 ``/public_dataset/1/ffdataset``。

    Args:
        path (str, os.PathLike): 数据集存放的主目录

    Examples:

        >>> hfai.datasets.set_data_dir("/your/data/dir")
        >>> hfai.datasets.get_data_dir()
        PosixPath('/your/data/dir')

    """
    global DATA_DIR
    DATA_DIR = Path(path).absolute()



def get_data_dir() -> Path:
    """
    返回当前数据集主目录

    Returns:
        data_dir (Path): 当前数据集主目录

    Examples:

        >>> hfai.datasets.set_data_dir("/your/data/dir")
        >>> hfai.datasets.get_data_dir()
        PosixPath('/your/data/dir')

    """
    global DATA_DIR

    # 1. set_data_dir() 设置的路径
    if DATA_DIR is not None:
        return DATA_DIR.absolute()

    # 2. 环境变量 HFAI_DATASETS_DIR 指定的路径
    env = os.getenv("HFAI_DATASETS_DIR")
    if env is not None:
        return Path(env)

    # 3. 默认路径
    return DEFAULT_DATA_DIR



class BaseDataset(Dataset):
    """hfai.dataset 基类 """

    _repr_indent = 4

    def __init__(self):
        """ """
        pass

    def __len__(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "hfai.datasets." + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if hasattr(self, "split") and self.split is not None:
            body.append(f"Split: {self.split}")

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def loader(self, *args, **kwargs) -> DataLoader:
        """
        获取数据集的Dataloader

        参数与 `PyTorch的Dataloader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ 保持一致

        Returns:
            数据集的Dataloader
        """

        return DataLoader(self, *args, **kwargs)



DATASETS = {}


def register_dataset(cls):
    if not issubclass(cls, BaseDataset):
        raise TypeError("Can only register classes inherited from BaseDataset")
    DATASETS[cls.__name__] = cls
    return cls