import numpy as np
from numpy.typing import NDArray


def add_awgn(sinal_entrada: NDArray[np.complex128], snr_db: float, ruido: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Adiciona ruído AWGN (Additive White Gaussian Noise) a um sinal.
    
    AWGN é o modelo mais comum de ruído em comunicações digitais:
    - Aditivo: o ruído é somado ao sinal
    - Branco: tem espectro plano (mesma potência em todas as frequências)
    - Gaussiano: amplitude segue distribuição normal
    
    Processo:
    1. Calcula potência média do sinal
    2. Converte SNR de dB para escala linear
    3. Calcula potência do ruído necessária para atingir o SNR desejado
    4. Gera ruído gaussiano com a potência calculada
    5. Adiciona ruído ao sinal
    
    Parâmetros:
        signal: sinal a ser corrompido (pode ser real ou complexo)
        snr_db: relação sinal-ruído em decibéis (SNR = potência_sinal / potência_ruído)
    
    Retorna:
        Sinal original com ruído adicionado
    """
    # Converter SNR de decibéis para escala linear
    # SNR_linear = 10^(SNR_dB / 10)
    snr_linear: np.float64 = 10.0 ** (snr_db / 10.0)

    # Calcular potência média do sinal
    # Para sinais complexos: |sinal|^2 = real^2 + imag^2
    # Para sinais reais: |sinal|^2 = sinal^2
    potencia_sinal: np.float64 = np.mean(np.power(np.abs(sinal_entrada, dtype=np.float64), 2, dtype=np.float64), dtype=np.float64)
    
    # Caso especial: sinal com potência zero (sinal nulo)
    if potencia_sinal == 0:
        tipo_ruido = np.complex128 if np.iscomplexobj(sinal_entrada) else np.float64
        ruido_zero = np.zeros_like(sinal_entrada, dtype=tipo_ruido)
        return sinal_entrada + ruido_zero

    # Calcular potência do ruído necessária
    # SNR = P_sinal / P_ruido  =>  P_ruido = P_sinal / SNR
    potencia_ruido = potencia_sinal / snr_linear

    # Gerar ruído gaussiano com a potência calculada
    # Para sinais complexos: ruído tem parte real e imaginária independentes
    # Cada parte tem metade da potência total
    desvio_padrao_ruido: np.complex128 = np.sqrt(np.complex128(potencia_ruido / 2.0), dtype=np.complex128)

    # Adicionar ruído ao sinal original
    sinal_com_ruido = sinal_entrada + ruido[:sinal_entrada.shape[0]] * desvio_padrao_ruido

    return sinal_com_ruido
