# ============================================================
# IMAGE INPUT EXPORT (matches notebook's preprocessing exactly)
# Uses the same transform pipeline as train_dataset/test_dataset:
#   ToTensor() + Normalize((0.1307,), (0.3081,))
# then the same int8 quantization used in int8_forward:
#   x = round(x * 32), clamped to int8 range
# ============================================================

from PIL import Image

IMG_PATH = "your_image.png"      # <-- set your image path
MEM_FILE = "nova_memory.mem"
IMG_START_ADDR = 0x00004500

# Same transform as used for train_dataset / test_dataset, with a resize
# added up front since MNIST images are natively 28x28 and yours may not be
img_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_and_quantize_image(path, transform):
    img = Image.open(path).convert("L")   # grayscale
    x = transform(img)                     # [1, 28, 28], normalized like MNIST

    # Same quantization as int8_forward: x = round(x * 32), clamp to int8
    q = torch.round(x * 32)
    q = torch.clamp(q, -128, 127).to(torch.int8)

    return q.squeeze(0).cpu().numpy()      # [28, 28] int8

def flatten_column_major(mat):
    """mat: [H, W] -> flatten column by column (top-to-bottom within each column)"""
    return mat.T.flatten().tolist()

def append_mem_dump(byte_list, filename, start_addr, bytes_per_block=16):
    with open(filename, "a") as f:
        for i in range(0, len(byte_list), bytes_per_block):
            block_addr = start_addr + i
            f.write(f"// 0x{block_addr:08X}\n")
            block = byte_list[i:i + bytes_per_block]
            for b in block:
                f.write(f"{to_u8(b):02X}\n")
    print(f"Appended {filename} : {len(byte_list)} bytes starting at "
          f"0x{start_addr:08X}, ending at 0x{start_addr + len(byte_list) - 1:08X}")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

img_q = load_and_quantize_image(IMG_PATH, img_transform)     # [28, 28] int8
img_bytes = flatten_column_major(img_q)                      # 784 values, column-major

pad = (-len(img_bytes)) % 16
if pad:
    img_bytes.extend([0] * pad)

append_mem_dump(img_bytes, MEM_FILE, IMG_START_ADDR)