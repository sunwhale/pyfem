from typing import Dict


class Properties:
    def __init__(self, dictionary: Dict = None):
        self.__dict__.update(dictionary or {})

    def __str__(self):
        props_list = []
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            props_list.append(f'{key}: {value}')
        return '\n'.join(props_list)

    def __iter__(self):
        return iter(self.__dict__.items())

    def store(self, key: str, val: object) -> None:
        """
        store 方法用于动态添加属性和值。如果属性名中包含点号 .，则表示这是一个嵌套属性名，需要在类的 __dict__ 属性中按照层级结构创建一个字典对象，并在最终嵌套层级上设置属性值。
        如果属性名不包含点号，则直接在类的 __dict__ 属性中添加属性和值。
        注意，在 store 方法中使用了字典的 setdefault 方法，当属性名不存在时，会创建一个空的字典作为属性值。
        """
        if "." in key:
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = obj.__dict__.setdefault(k, {})
            obj.__dict__[keys[-1]] = clean_variable(val)
        else:
            self.__dict__[key] = clean_variable(val)


if __name__ == "__main__":
    props = Properties()

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    from pprint import pprint

    def read_toml(file_name: str) -> None:
        with open(file_name, "rb") as f:
            toml = tomllib.load(f)
        return toml

    props_dict =  read_toml(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    for key, val in props_dict.items():
        props.store(key, val)

    print(props.mesh)