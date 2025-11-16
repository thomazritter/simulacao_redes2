
import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
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
    # Garantir que o sinal seja um array numpy
    sinal_entrada = np.asarray(signal)
    
    # Converter SNR de decibéis para escala linear
    # SNR_linear = 10^(SNR_dB / 10)
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Calcular potência média do sinal
    # Para sinais complexos: |sinal|^2 = real^2 + imag^2
    # Para sinais reais: |sinal|^2 = sinal^2
    potencia_sinal = np.mean(np.abs(sinal_entrada) ** 2)
    
    # Caso especial: sinal com potência zero (sinal nulo)
    if potencia_sinal == 0:
        # Retornar sinal sem ruído (não faz sentido adicionar ruído a sinal nulo)
        tipo_ruido = np.complex128 if np.iscomplexobj(sinal_entrada) else np.float64
        ruido_zero = np.zeros_like(sinal_entrada, dtype=tipo_ruido)
        return sinal_entrada + ruido_zero

    # Calcular potência do ruído necessária
    # SNR = P_sinal / P_ruido  =>  P_ruido = P_sinal / SNR
    potencia_ruido = potencia_sinal / snr_linear

    # Gerar ruído gaussiano com a potência calculada
    if np.iscomplexobj(sinal_entrada):
        # Para sinais complexos: ruído tem parte real e imaginária independentes
        # Cada parte tem metade da potência total
        desvio_padrao_ruido = np.sqrt(potencia_ruido / 2.0)
        # Gerar ruído gaussiano complexo
        ruido_real = np.random.randn(*sinal_entrada.shape)
        ruido_imag = np.random.randn(*sinal_entrada.shape)
        ruido = desvio_padrao_ruido * (ruido_real + 1j * ruido_imag)
    else:
        # Para sinais reais: ruído é apenas parte real
        desvio_padrao_ruido = np.sqrt(potencia_ruido)
        # Gerar ruído gaussiano real
        ruido = desvio_padrao_ruido * np.random.randn(*sinal_entrada.shape)

    # Adicionar ruído ao sinal original
    sinal_com_ruido = sinal_entrada + ruido

    # Mostrar efeito do ruído AWGN
    amostra = min(8, len(sinal_entrada))
    if np.iscomplexobj(sinal_entrada):
        antes_str = ' '.join(f'{s.real:+.2f}{s.imag:+.2f}j' for s in sinal_entrada[:amostra])
        depois_str = ' '.join(f'{s.real:+.2f}{s.imag:+.2f}j' for s in sinal_com_ruido[:amostra])
    else:
        antes_str = ' '.join(f'{s:+.2f}' for s in sinal_entrada[:amostra])
        depois_str = ' '.join(f'{s:+.2f}' for s in sinal_com_ruido[:amostra])
    if len(sinal_entrada) > amostra:
        antes_str += "..."
        depois_str += "..."
    print(f"  [4. CANAL AWGN (SNR = {snr_db:.1f} dB)]")
    print(f"     Antes:  [{antes_str}] (potência: {potencia_sinal:.4f})")
    print(f"     Depois: [{depois_str}] (potência ruído: {potencia_ruido:.4f})")

    return sinal_com_ruido
