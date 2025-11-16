
from typing import Tuple
import numpy as np


def pulse_shape(symbols: np.ndarray, samples_per_symbol: int, rolloff: float = 0.35) -> np.ndarray:
    """
    Aplica formatação de pulso (pulse shaping) usando filtro raised cosine.
    
    Em sistemas reais, cada símbolo não é um degrau, mas sim um pulso formatado
    para reduzir interferência entre símbolos (ISI) e limitar largura de banda.
    
    Parâmetros:
        symbols: símbolos a formatar
        samples_per_symbol: número de amostras por símbolo
        rolloff: fator de rolloff do filtro (0.0 = retangular, 0.35 = típico)
    
    Retorna:
        Sinal formatado com pulse shaping
    """
    if rolloff == 0.0:
        # Pulso retangular (simples, mas não ideal)
        return np.repeat(symbols, samples_per_symbol)
    
    # Filtro raised cosine (mais realista)
    # Para simplificar, usamos interpolação com filtro passa-baixa
    # Em sistemas reais, usaria convolução com resposta ao impulso do raised cosine
    num_symbols = len(symbols)
    num_samples = num_symbols * samples_per_symbol
    
    # Interpolar símbolos (upsampling)
    indices_symbols = np.arange(num_symbols) * samples_per_symbol
    indices_samples = np.arange(num_samples)
    
    # Interpolação linear (aproximação simples do pulse shaping)
    # Em sistemas reais, usaria convolução com resposta ao impulso do filtro
    if np.iscomplexobj(symbols):
        real_part = np.interp(indices_samples, indices_symbols, np.real(symbols))
        imag_part = np.interp(indices_samples, indices_symbols, np.imag(symbols))
        return real_part + 1j * imag_part
    else:
        return np.interp(indices_samples, indices_symbols, symbols)


def add_carrier(symbols: np.ndarray, fc: float, fs: float, t_start: float = 0.0, use_pulse_shaping: bool = True) -> np.ndarray:
    """
    Adiciona portadora ao sinal em banda base (modulação passa-banda).
    
    Para BPSK: multiplica símbolos reais por cos(2πfc*t)
    Para QPSK: multiplica símbolos complexos usando componentes I e Q
    
    Processo:
    1. Aplica pulse shaping (formatação de pulso) - mais realista
    2. Gera vetor de tempo para cada símbolo
    3. Para BPSK: sinal(t) = símbolo_formatado * cos(2πfc*t)
    4. Para QPSK: sinal(t) = I*cos(2πfc*t) - Q*sin(2πfc*t)
    
    IMPORTANTE: Em sistemas reais, fc >> taxa_de_símbolos
    (frequência da portadora muito maior que taxa de símbolos)
    
    Parâmetros:
        symbols: símbolos em banda base (reais para BPSK, complexos para QPSK)
        fc: frequência da portadora (Hz) - deve ser >> 1/duração_símbolo
        fs: taxa de amostragem (amostras por símbolo)
        t_start: tempo inicial (segundos)
        use_pulse_shaping: se True, aplica formatação de pulso (mais realista)
    
    Retorna:
        Sinal modulado em passa-banda (valores reais)
    """
    simbolos = np.asarray(symbols)
    num_simbolos = len(simbolos)
    
    # Aplicar pulse shaping (formatação de pulso) - mais realista
    if use_pulse_shaping:
        simbolos_formatados = pulse_shape(simbolos, int(fs), rolloff=0.35)
    else:
        # Pulso retangular (menos realista, mas mais simples)
        simbolos_formatados = np.repeat(simbolos, int(fs))
    
    num_amostras = len(simbolos_formatados)
    
    # Gerar vetor de tempo
    # Duração total = num_simbolos períodos de símbolo
    t = np.linspace(t_start, t_start + num_simbolos, num_amostras, endpoint=False)
    
    # Gerar portadora
    # Em sistemas reais: fc >> 1 (frequência da portadora muito maior que taxa de símbolos)
    portadora_cos = np.cos(2 * np.pi * fc * t)
    portadora_sin = np.sin(2 * np.pi * fc * t)
    
    if np.iscomplexobj(simbolos_formatados):
        # QPSK: usar componentes I (real) e Q (imaginário)
        # Modulação passa-banda: I*cos - Q*sin
        sinal_passabanda = np.real(simbolos_formatados) * portadora_cos - np.imag(simbolos_formatados) * portadora_sin
    else:
        # BPSK: apenas componente I
        # Modulação passa-banda: símbolo * cos
        sinal_passabanda = simbolos_formatados * portadora_cos
    
    # Mostrar efeito da portadora
    amostra = min(8, len(simbolos))
    if np.iscomplexobj(simbolos):
        simbolos_str = ' '.join(f'{s.real:+.2f}{s.imag:+.2f}j' for s in simbolos[:amostra])
    else:
        simbolos_str = ' '.join(f'{s:+.2f}' for s in simbolos[:amostra])
    sinal_str = ' '.join(f'{s:+.2f}' for s in sinal_passabanda[:min(16, len(sinal_passabanda))])
    if len(simbolos) > amostra:
        simbolos_str += "..."
    if len(sinal_passabanda) > 16:
        sinal_str += "..."
    print(f"  [3.5. ADIÇÃO DE PORTADORA (fc={fc:.1f} Hz, fs={fs:.1f})]")
    print(f"     Símbolos banda base: [{simbolos_str}] (tamanho: {len(simbolos)})")
    print(f"     Sinal passa-banda: [{sinal_str}] (tamanho: {len(sinal_passabanda)})")
    
    return sinal_passabanda


def lowpass_filter(signal: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    """
    Filtro passa-baixa simples para remover componentes de alta frequência.
    
    Em sistemas reais, usaria filtros mais sofisticados (FIR/IIR),
    mas para simulação, média móvel é uma aproximação razoável.
    
    Parâmetros:
        signal: sinal a filtrar
        samples_per_symbol: número de amostras por símbolo
    
    Retorna:
        Sinal filtrado
    """
    # Filtro passa-baixa simples: média móvel
    # Em sistemas reais, usaria filtro matched filter ou raised cosine
    window_size = int(samples_per_symbol)
    if window_size < 2:
        return signal
    
    # Aplicar média móvel (aproximação de filtro passa-baixa)
    filtered = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    return filtered


def remove_carrier(signal: np.ndarray, fc: float, fs: float, t_start: float = 0.0, use_filtering: bool = True) -> np.ndarray:
    """
    Remove portadora do sinal passa-banda (demodulação para banda base).
    
    Processo:
    1. Multiplica sinal recebido por cos(2πfc*t) e sin(2πfc*t) (downconversion)
    2. Filtra passa-baixa para remover componentes de alta frequência
    3. Decima (downsample) para uma amostra por símbolo
    4. Reconstrói símbolos
    
    Em sistemas reais:
    - Usa filtros matched filter ou raised cosine
    - Requer sincronização precisa de fase e frequência
    - Pode ter erro de fase/frequência que precisa ser corrigido
    
    Parâmetros:
        signal: sinal em passa-banda (valores reais)
        fc: frequência da portadora (Hz)
        fs: taxa de amostragem (amostras por símbolo)
        t_start: tempo inicial (segundos)
        use_filtering: se True, aplica filtro passa-baixa (mais realista)
    
    Retorna:
        Símbolos demodulados em banda base
    """
    sinal = np.asarray(signal)
    num_amostras = len(sinal)
    num_simbolos = int(num_amostras / fs)
    
    # Gerar vetor de tempo
    t = np.linspace(t_start, t_start + num_simbolos, num_amostras, endpoint=False)
    
    # Gerar portadoras locais (downconversion)
    # Em sistemas reais, estas portadoras precisam estar sincronizadas
    # com a portadora do transmissor (PLL - Phase Locked Loop)
    portadora_cos = np.cos(2 * np.pi * fc * t)
    portadora_sin = np.sin(2 * np.pi * fc * t)
    
    # Demodulação: multiplicar por portadoras (downconversion)
    # Isso move o sinal de volta para banda base
    componente_i = sinal * portadora_cos
    componente_q = -sinal * portadora_sin  # Negativo para Q
    
    # Filtrar passa-baixa para remover componentes de alta frequência
    # (resultantes da multiplicação: 2fc, etc.)
    if use_filtering:
        componente_i = lowpass_filter(componente_i, int(fs))
        componente_q = lowpass_filter(componente_q, int(fs))
    
    # Decimar (downsample): uma amostra por símbolo
    # Em sistemas reais, isso é feito após o matched filter
    indices_amostragem = np.arange(0, num_amostras, int(fs))
    indices_amostragem = indices_amostragem[:num_simbolos]  # Garantir tamanho correto
    
    simbolos_i = componente_i[indices_amostragem]
    simbolos_q = componente_q[indices_amostragem]
    
    # Reconstruir símbolos complexos (ou reais para BPSK)
    simbolos_demodulados = simbolos_i + 1j * simbolos_q
    
    # Mostrar efeito da remoção de portadora
    amostra = min(8, len(simbolos_demodulados))
    simbolos_str = ' '.join(f'{s.real:+.2f}{s.imag:+.2f}j' for s in simbolos_demodulados[:amostra])
    if len(simbolos_demodulados) > amostra:
        simbolos_str += "..."
    amostra_sinal = min(16, len(sinal))
    sinal_str = ' '.join(f'{s:+.2f}' for s in sinal[:amostra_sinal])
    if len(sinal) > amostra_sinal:
        sinal_str += "..."
    print(f"  [5.5. REMOÇÃO DE PORTADORA (fc={fc:.1f} Hz)]")
    print(f"     Sinal passa-banda: [{sinal_str}] (tamanho: {len(sinal)})")
    print(f"     Símbolos banda base: [{simbolos_str}] (tamanho: {len(simbolos_demodulados)})")
    
    return simbolos_demodulados


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Modula bits usando BPSK (Binary Phase Shift Keying).
    
    BPSK é a modulação digital mais simples:
    - Bit 0 -> símbolo -1 (fase 180°)
    - Bit 1 -> símbolo +1 (fase 0°)
    
    Vantagens:
    - Simples de implementar
    - Robusta contra ruído
    
    Desvantagem:
    - Baixa eficiência espectral (1 bit por símbolo)
    
    Parâmetros:
        bits: array numpy com bits a modular (0 ou 1)
    
    Retorna:
        Array numpy com símbolos modulados (-1 ou +1)
    """
    # Garantir que bits seja um array numpy
    bits_entrada = np.asarray(bits, dtype=np.uint8)
    
    # Validar formato
    if bits_entrada.ndim != 1:
        raise ValueError("bits deve ser um vetor 1D")
    
    # Aplicar mapeamento BPSK:
    # Bit 0 -> -1: 2*0 - 1 = -1
    # Bit 1 -> +1: 2*1 - 1 = +1
    simbolos_modulados = 2.0 * bits_entrada.astype(np.float64) - 1.0
    
    # Mostrar conversão bits → símbolos BPSK
    amostra = min(16, len(bits_entrada))
    bits_str = ''.join(str(b) for b in bits_entrada[:amostra])
    simbolos_str = ' '.join(f'{s:+.1f}' for s in simbolos_modulados[:amostra])
    if len(bits_entrada) > amostra:
        bits_str += "..."
        simbolos_str += "..."
    print(f"  [3. MANCHESTER → MODULAÇÃO BPSK]")
    print(f"     Bits:    [{bits_str}] (tamanho: {len(bits_entrada)})")
    print(f"     Símbolos: [{simbolos_str}] (tamanho: {len(simbolos_modulados)})")
    
    return simbolos_modulados


def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    Demodula símbolos BPSK de volta para bits.
    
    Processo:
    - Símbolo >= 0 (positivo ou zero) -> Bit 1
    - Símbolo < 0 (negativo) -> Bit 0
    
    Esta é uma decisão por limiar simples no zero.
    
    Parâmetros:
        symbols: array numpy com símbolos modulados (valores reais)
    
    Retorna:
        Array numpy com bits demodulados (0 ou 1)
    """
    # Garantir que symbols seja um array numpy
    simbolos_entrada = np.asarray(symbols, dtype=np.float64)
    
    # Decisão por limiar:
    # Se símbolo >= 0: decidir por bit 1
    # Se símbolo < 0: decidir por bit 0
    bits_demodulados = (simbolos_entrada >= 0).astype(np.uint8)
    
    # Mostrar conversão símbolos → bits (demodulação BPSK)
    amostra = min(16, len(simbolos_entrada))
    simbolos_str = ' '.join(f'{s:+.2f}' for s in simbolos_entrada[:amostra])
    bits_str = ''.join(str(b) for b in bits_demodulados[:amostra])
    if len(simbolos_entrada) > amostra:
        simbolos_str += "..."
        bits_str += "..."
    print(f"  [5. DEMODULAÇÃO BPSK → MANCHESTER]")
    print(f"     Símbolos: [{simbolos_str}] (tamanho: {len(simbolos_entrada)})")
    print(f"     Bits:    [{bits_str}] (tamanho: {len(bits_demodulados)})")
    
    return bits_demodulados


def qpsk_modulate(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Modula bits usando QPSK (Quadrature Phase Shift Keying) com mapeamento Gray.
    
    QPSK transmite 2 bits por símbolo, dobrando a eficiência espectral:
    - (0,0) -> símbolo (1+1j) / sqrt(2)  [fase 45°]
    - (0,1) -> símbolo (-1+1j) / sqrt(2)  [fase 135°]
    - (1,1) -> símbolo (-1-1j) / sqrt(2)  [fase 225°]
    - (1,0) -> símbolo (1-1j) / sqrt(2)   [fase 315°]
    
    Mapeamento Gray: símbolos adjacentes diferem em apenas 1 bit
    (reduz erros quando há pequenos erros de fase)
    
    Normalização: divide por sqrt(2) para ter potência média = 1
    
    Parâmetros:
        bits: array numpy com bits a modular (0 ou 1)
    
    Retorna:
        (símbolos_modulados, padding_bits)
        - símbolos_modulados: array complexo com símbolos QPSK
        - padding_bits: número de bits de preenchimento adicionados (0 ou 1)
    """
    # Garantir que bits seja um array numpy
    bits_entrada = np.asarray(bits, dtype=np.uint8)
    
    # Validar formato
    if bits_entrada.ndim != 1:
        raise ValueError("bits deve ser um vetor 1D")

    # QPSK precisa de número par de bits (2 bits por símbolo)
    # Se número ímpar, adicionar um bit zero no final (padding)
    numero_bits_originais = len(bits_entrada)
    padding_bits = 0
    
    if numero_bits_originais % 2 != 0:
        # Adicionar um bit zero no final
        bits_entrada = np.concatenate([bits_entrada, np.zeros(1, dtype=np.uint8)])
        padding_bits = 1

    # Agrupar bits em pares (cada par vira um símbolo QPSK)
    pares_de_bits = bits_entrada.reshape(-1, 2)
    numero_simbolos = len(pares_de_bits)
    
    # Criar array para símbolos modulados (complexos)
    simbolos_modulados = np.empty(numero_simbolos, dtype=np.complex128)

    # Mapear cada par de bits para um símbolo QPSK (mapeamento Gray)
    for indice, (bit0, bit1) in enumerate(pares_de_bits):
        if bit0 == 0 and bit1 == 0:
            # (0,0) -> fase 45° (quadrante I)
            simbolos_modulados[indice] = 1 + 1j
        elif bit0 == 0 and bit1 == 1:
            # (0,1) -> fase 135° (quadrante II)
            simbolos_modulados[indice] = -1 + 1j
        elif bit0 == 1 and bit1 == 1:
            # (1,1) -> fase 225° (quadrante III)
            simbolos_modulados[indice] = -1 - 1j
        elif bit0 == 1 and bit1 == 0:
            # (1,0) -> fase 315° (quadrante IV)
            simbolos_modulados[indice] = 1 - 1j
        else:
            raise ValueError("bits devem ser 0 ou 1")

    # Normalizar potência média para 1
    # Cada símbolo tem magnitude sqrt(2), então dividir por sqrt(2) normaliza
    fator_normalizacao = np.sqrt(2.0)
    simbolos_modulados /= fator_normalizacao
    
    # Mostrar conversão bits → símbolos QPSK
    amostra = min(8, len(pares_de_bits))
    bits_str = ' '.join(f'{b0}{b1}' for b0, b1 in pares_de_bits[:amostra])
    simbolos_str = ' '.join(f'{s.real:+.2f}{s.imag:+.2f}j' for s in simbolos_modulados[:amostra])
    if len(pares_de_bits) > amostra:
        bits_str += "..."
        simbolos_str += "..."
    print(f"  [3. MANCHESTER → MODULAÇÃO QPSK]")
    print(f"     Bits (pares): [{bits_str}] (tamanho: {len(bits_entrada)})")
    print(f"     Símbolos: [{simbolos_str}] (tamanho: {len(simbolos_modulados)})")
    
    return simbolos_modulados, padding_bits


def qpsk_demodulate(symbols: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Demodula símbolos QPSK de volta para bits.
    
    Processo:
    1. Desnormaliza símbolos (multiplica por sqrt(2))
    2. Para cada símbolo, verifica sinal da parte real e imaginária
    3. Mapeia quadrante para par de bits (inverso do mapeamento Gray)
    
    Decisão por quadrante:
    - Quadrante I (Re>=0, Im>=0)   -> (0,0)
    - Quadrante II (Re<0, Im>=0)    -> (0,1)
    - Quadrante III (Re<0, Im<0)    -> (1,1)
    - Quadrante IV (Re>=0, Im<0)    -> (1,0)
    
    Parâmetros:
        symbols: array numpy com símbolos modulados (valores complexos)
        padding: número de bits de preenchimento a remover (0 ou 1)
    
    Retorna:
        Array numpy com bits demodulados (0 ou 1)
    """
    # Garantir que symbols seja um array numpy
    simbolos_entrada = np.asarray(symbols, dtype=np.complex128)
    
    # Desnormalizar símbolos (reverter divisão por sqrt(2))
    fator_normalizacao = np.sqrt(2.0)
    simbolos_desnormalizados = simbolos_entrada * fator_normalizacao

    # Lista para armazenar bits demodulados
    bits_demodulados = []
    
    # Para cada símbolo, decidir qual par de bits foi transmitido
    for simbolo in simbolos_desnormalizados:
        # Extrair parte real e imaginária
        parte_real = simbolo.real
        parte_imag = simbolo.imag
        
        # Decisão por quadrante (inverso do mapeamento Gray)
        if parte_real >= 0 and parte_imag >= 0:
            # Quadrante I -> (0,0)
            bits_demodulados.extend([0, 0])
        elif parte_real < 0 and parte_imag >= 0:
            # Quadrante II -> (0,1)
            bits_demodulados.extend([0, 1])
        elif parte_real < 0 and parte_imag < 0:
            # Quadrante III -> (1,1)
            bits_demodulados.extend([1, 1])
        else:  # parte_real >= 0 and parte_imag < 0
            # Quadrante IV -> (1,0)
            bits_demodulados.extend([1, 0])

    # Remover bits de preenchimento se foram adicionados durante modulação
    if padding > 0:
        bits_demodulados = bits_demodulados[:-padding]
    
    # Converter lista para array numpy
    bits_array = np.array(bits_demodulados, dtype=np.uint8)
    
    # Mostrar conversão símbolos → bits (demodulação QPSK)
    amostra = min(8, len(simbolos_desnormalizados))
    simbolos_str = ' '.join(f'{s.real:+.2f}{s.imag:+.2f}j' for s in simbolos_desnormalizados[:amostra])
    bits_str = ''.join(str(b) for b in bits_array[:min(16, len(bits_array))])
    if len(simbolos_desnormalizados) > amostra:
        simbolos_str += "..."
    if len(bits_array) > 16:
        bits_str += "..."
    print(f"  [5. DEMODULAÇÃO QPSK → MANCHESTER]")
    print(f"     Símbolos: [{simbolos_str}] (tamanho: {len(simbolos_desnormalizados)})")
    print(f"     Bits:    [{bits_str}] (tamanho: {len(bits_array)})")
    
    return bits_array
