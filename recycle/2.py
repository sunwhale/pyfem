import os
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


def split_and_encrypt_doc(input_file, n_parts, password):
    """
    拆分并加密.DOC文件

    参数:
    input_file: 输入的.DOC文件路径
    n_parts: 拆分成多少部分
    password: 加密密码
    """

    # 读取原始文件
    with open(input_file, 'rb') as f:
        file_data = f.read()

    file_size = len(file_data)
    print(f"原始文件大小: {file_size} 字节")

    # 计算每部分大小
    part_size = file_size // n_parts
    remainder = file_size % n_parts

    # 生成密钥
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    cipher = Fernet(key)

    # 创建输出目录
    output_dir = "encrypted_parts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存密钥和元数据
    metadata = {
        'original_filename': os.path.basename(input_file),
        'file_size': file_size,
        'n_parts': n_parts,
        'salt': base64.b64encode(salt).decode('utf-8')
    }

    # 拆分并加密文件
    start_index = 0
    encrypted_parts = []

    for i in range(n_parts):
        # 计算当前部分大小（最后一部分包含余数）
        current_part_size = part_size + (remainder if i == n_parts - 1 else 0)

        # 提取部分数据
        part_data = file_data[start_index:start_index + current_part_size]

        # 加密数据
        encrypted_data = cipher.encrypt(part_data)

        # 生成文件名
        part_filename = os.path.join(output_dir, f"part_{i + 1:03d}.enc")

        # 写入加密部分
        with open(part_filename, 'wb') as f:
            # 写入部分头信息（部分编号和大小）
            header = f"PART:{i + 1}/{n_parts}|SIZE:{len(encrypted_data)}\n".encode('utf-8')
            f.write(header)
            f.write(encrypted_data)

        encrypted_parts.append({
            'filename': part_filename,
            'original_size': len(part_data),
            'encrypted_size': len(encrypted_data)
        })

        print(f"创建部分 {i + 1}/{n_parts}: {len(part_data)} 字节 -> {len(encrypted_data)} 字节")

        start_index += current_part_size

    # 保存元数据文件
    metadata_file = os.path.join(output_dir, "metadata.enc")
    with open(metadata_file, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)

    print(f"\n拆分完成！")
    print(f"原始文件: {input_file}")
    print(f"拆分成: {n_parts} 个部分")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_file}")
    print(f"请妥善保管密码！")

    return output_dir, metadata


if __name__ == "__main__":
    # 使用示例
    input_doc = "1.docx"  # 替换为您的.DOC文件路径
    n_parts = 29  # 拆分成5部分
    password = "MySecurePassword123!"  # 设置加密密码

    if os.path.exists(input_doc):
        split_and_encrypt_doc(input_doc, n_parts, password)
    else:
        print(f"错误: 文件 {input_doc} 不存在！")
        print("请将要处理的.DOC文件放在当前目录，并修改脚本中的文件名。")