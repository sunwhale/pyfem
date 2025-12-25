import os
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def decrypt_and_merge_doc(encrypted_dir, password, output_filename=None):
    """
    解密并合并.DOC文件

    参数:
    encrypted_dir: 加密部分的目录
    password: 解密密码
    output_filename: 输出文件名（可选）
    """

    # 读取元数据
    metadata_file = os.path.join(encrypted_dir, "metadata.enc")

    if not os.path.exists(metadata_file):
        print(f"错误: 在 {encrypted_dir} 中找不到元数据文件！")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    original_filename = metadata['original_filename']
    file_size = metadata['file_size']
    n_parts = metadata['n_parts']
    salt = base64.b64decode(metadata['salt'])

    # 生成密钥
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    cipher = Fernet(key)

    # 收集所有部分文件
    parts = []
    for i in range(1, n_parts + 1):
        part_filename = os.path.join(encrypted_dir, f"part_{i:03d}.enc")
        if os.path.exists(part_filename):
            parts.append(part_filename)
        else:
            print(f"警告: 找不到部分文件 {part_filename}")

    if len(parts) != n_parts:
        print(f"错误: 找到 {len(parts)} 个部分，但需要 {n_parts} 个部分！")
        return

    # 解密并合并数据
    decrypted_data = bytearray()

    for i, part_file in enumerate(parts, 1):
        with open(part_file, 'rb') as f:
            # 读取头信息
            header_line = f.readline().decode('utf-8').strip()
            if not header_line.startswith("PART:"):
                print(f"错误: {part_file} 的格式不正确！")
                return

            # 读取加密数据
            encrypted_data = f.read()

            try:
                # 解密数据
                decrypted_part = cipher.decrypt(encrypted_data)
                decrypted_data.extend(decrypted_part)

                print(f"解密部分 {i}/{n_parts}: {len(encrypted_data)} 字节 -> {len(decrypted_part)} 字节")
            except Exception as e:
                print(f"错误: 解密部分 {i} 失败！")
                print(f"可能密码不正确或文件已损坏。")
                print(f"错误信息: {str(e)}")
                return

    # 验证数据大小
    if len(decrypted_data) != file_size:
        print(f"警告: 解密后数据大小 ({len(decrypted_data)} 字节) 与原始大小 ({file_size} 字节) 不匹配！")

    # 确定输出文件名
    if output_filename is None:
        # 在原始文件名后添加 "_restored"
        name, ext = os.path.splitext(original_filename)
        output_filename = f"{name}_restored{ext}"

    # 写入恢复的文件
    with open(output_filename, 'wb') as f:
        f.write(decrypted_data)

    print(f"\n恢复完成！")
    print(f"输出文件: {output_filename}")
    print(f"文件大小: {len(decrypted_data)} 字节")

    # 验证文件完整性
    try:
        # 简单的完整性检查：比较前100字节的哈希
        sample_hash = hashlib.md5(decrypted_data[:min(1000, len(decrypted_data))]).hexdigest()
        print(f"文件完整性检查: 前1000字节的MD5哈希: {sample_hash}")
    except:
        pass

    return output_filename


def find_encrypted_directories():
    """查找可能的加密目录"""
    dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item):
            metadata_file = os.path.join(item, "metadata.enc")
            if os.path.exists(metadata_file):
                dirs.append(item)
    return dirs


if __name__ == "__main__":
    # 使用示例
    encrypted_dirs = find_encrypted_directories()

    if encrypted_dirs:
        print("找到以下加密目录:")
        for i, dir_name in enumerate(encrypted_dirs, 1):
            print(f"  {i}. {dir_name}")

        dir_choice = input(f"\n请选择要解密的目录 (1-{len(encrypted_dirs)}): ").strip()

        try:
            dir_index = int(dir_choice) - 1
            if 0 <= dir_index < len(encrypted_dirs):
                selected_dir = encrypted_dirs[dir_index]
                password = input("请输入解密密码: ").strip()

                output_name = input("请输入输出文件名（按Enter使用默认名称）: ").strip()
                if not output_name:
                    output_name = None

                decrypt_and_merge_doc(selected_dir, password, output_name)
            else:
                print("无效的选择！")
        except ValueError:
            print("请输入有效的数字！")
    else:
        # 手动指定目录
        encrypted_dir = "encrypted_parts"  # 替换为您的加密目录
        password = "MySecurePassword123!"  # 替换为您的密码

        if os.path.exists(encrypted_dir):
            decrypt_and_merge_doc(encrypted_dir, password)
        else:
            print(f"错误: 目录 {encrypted_dir} 不存在！")
            print("请确保加密目录在当前目录下。")