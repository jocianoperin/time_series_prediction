"""
Utilitário para liberar completamente a VRAM entre execuções
Compatível com CUDA 11+; dispensa dependências externas.
"""
import ctypes
import gc
import tensorflow as tf

def free_gpu_memory(verbose: bool = False):
    """Libera grafo TF, força GC e (se houver contexto) reseta a GPU."""
    tf.keras.backend.clear_session()
    gc.collect()

    # tenta resetar apenas se um contexto foi criado
    if not tf.config.list_physical_devices("GPU"):
        return

    try:
        libcudart = ctypes.CDLL("libcudart.so")
        status = libcudart.cudaDeviceReset()
        # 0 = sucesso | 3 = device NÃO inicializado => ok ignorar
        if status not in (0, 3) and verbose:
            print(f"[GPU_UTILS] cudaDeviceReset() retornou {status}")
    except OSError:
        # libcudart não disponível (CPU‑only) – simplesmente ignore
        pass
