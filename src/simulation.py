
from typing import List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

from .encoding import text_to_bits, bits_to_text, manchester_encode, manchester_decode
from .modulation import (
    bpsk_modulate,
    bpsk_demodulate,
    qpsk_modulate,
    qpsk_demodulate,
    add_carrier,
    remove_carrier,
)
from .channel import add_awgn


def bit_error_rate(original: np.ndarray, received: np.ndarray) -> Tuple[float, int, int]:
    """
    Calcula a Taxa de Erro de Bits (BER) comparando bits originais com bits recebidos.
    
    Passos:
    1. Garante que ambos os vetores tenham o mesmo tamanho
    2. Compara bit a bit para encontrar erros
    3. Calcula BER = número de erros / total de bits
    
    Retorna:
        (BER, número_de_erros, total_de_bits)
    """
    # Garantir que ambos os vetores tenham o mesmo tamanho
    tamanho_original = len(original)
    tamanho_recebido = len(received)
    tamanho_comparacao = min(tamanho_original, tamanho_recebido)
    
    if tamanho_comparacao == 0:
        return (0.0, 0, 0)
    
    # Converter para o mesmo tipo para comparação
    bits_originais = original[:tamanho_comparacao].astype(np.uint8)
    bits_recebidos = received[:tamanho_comparacao].astype(np.uint8)
    
    # Comparar bit a bit: True onde há diferença (erro)
    bits_com_erro = (bits_originais != bits_recebidos)
    numero_de_erros = np.sum(bits_com_erro)
    
    # Calcular BER: proporção de bits errados
    ber = numero_de_erros / float(tamanho_comparacao)
    
    return (ber, numero_de_erros, tamanho_comparacao)


def simulate_ber_bpsk(bits: np.ndarray, snr_db_values: List[float], use_manchester: bool = True, use_carrier: bool = False, fc: float = 1.0, fs: float = 10.0) -> List[float]:
    """
    Simula a Taxa de Erro de Bits (BER) para modulação BPSK.
    
    Fluxo do processo:
    1. Codificação (opcional): aplica Manchester encoding se solicitado
    2. Modulação: converte bits em símbolos BPSK (0 -> -1, 1 -> +1)
    2.5. Portadora (opcional): adiciona portadora para modulação passa-banda
    3. Conversão SNR: transforma Eb/N0 (energia por bit) em Es/N0 (energia por símbolo)
    4. Canal: adiciona ruído AWGN ao sinal
    4.5. Remoção portadora (opcional): remove portadora se foi adicionada
    5. Demodulação: converte símbolos de volta para bits
    6. Decodificação (opcional): remove Manchester encoding se foi aplicado
    7. Cálculo BER: compara bits recebidos com bits originais
    
    Parâmetros:
        bits: bits de informação originais
        snr_db_values: lista de valores de SNR em dB (Eb/N0)
        use_manchester: se True, aplica codificação Manchester
        use_carrier: se True, adiciona portadora (modulação passa-banda)
        fc: frequência da portadora em Hz (padrão: 1.0)
        fs: taxa de amostragem (amostras por símbolo) (padrão: 10.0)
    
    Retorna:
        Lista de valores de BER para cada SNR
    """
    # PASSO 1: Preparar bits para transmissão
    bits_originais = bits.copy()
    bits_para_transmitir = bits_originais
    
    # Aplicar codificação Manchester se solicitado
    if use_manchester:
        bits_para_transmitir = manchester_encode(bits_originais)
        print(f"  [BPSK] Bits originais: {len(bits_originais)}, após Manchester: {len(bits_para_transmitir)}")

    # Lista para armazenar resultados de BER
    resultados_ber = []
    
    # Para cada valor de SNR, realizar simulação
    for snr_eb_n0_db in snr_db_values:
        print(f"\n  --- Simulando BPSK com SNR = {snr_eb_n0_db:.1f} dB ---")
        
        # PASSO 2: Modulação BPSK
        # Converte bits (0/1) em símbolos (-1/+1)
        simbolos_banda_base = bpsk_modulate(bits_para_transmitir)
        
        # PASSO 2.5: Adicionar portadora (opcional)
        # Em sistemas reais: fc >> taxa_de_símbolos (portadora muito maior)
        # Aplica pulse shaping para tornar mais realista
        if use_carrier:
            sinal_transmitido = add_carrier(simbolos_banda_base, fc, fs, use_pulse_shaping=True)
        else:
            sinal_transmitido = simbolos_banda_base
        
        # PASSO 3: Conversão de Eb/N0 para Es/N0
        # Eb/N0 = energia por bit de informação
        # Es/N0 = energia por símbolo modulado
        # 
        # Com Manchester: cada bit de informação vira 2 símbolos BPSK
        # Portanto: Es = Eb/2, então Es/N0 = Eb/N0 - 10*log10(2) ≈ Eb/N0 - 3 dB
        #
        # Sem Manchester: cada bit vira 1 símbolo, então Es/N0 = Eb/N0
        if use_manchester:
            # Cada bit de informação vira 2 símbolos, então a energia por símbolo é menor
            fator_manchester_db = 10 * np.log10(2)  # ≈ 3.01 dB
            snr_es_n0_db = snr_eb_n0_db - fator_manchester_db
        else:
            # Sem Manchester: 1 bit = 1 símbolo, sem ajuste necessário
            snr_es_n0_db = snr_eb_n0_db
        
        # PASSO 4: Simular canal com ruído AWGN
        # Adiciona ruído gaussiano branco ao sinal transmitido
        sinal_com_ruido = add_awgn(sinal_transmitido, snr_es_n0_db)
        
        # PASSO 4.5: Remover portadora (se foi adicionada)
        # Em sistemas reais, requer sincronização precisa (PLL - Phase Locked Loop)
        # Aplica filtro passa-baixa para remover componentes de alta frequência
        if use_carrier:
            simbolos_recebidos = remove_carrier(sinal_com_ruido, fc, fs, use_filtering=True)
        else:
            simbolos_recebidos = sinal_com_ruido
        
        # PASSO 5: Demodulação BPSK
        # Converte símbolos recebidos de volta para bits
        # Decisão: símbolo >= 0 -> bit 1, símbolo < 0 -> bit 0
        # Para BPSK, usar apenas parte real se for complexo (após remoção de portadora)
        if np.iscomplexobj(simbolos_recebidos):
            simbolos_recebidos = np.real(simbolos_recebidos)
        bits_demodulados = bpsk_demodulate(simbolos_recebidos)
        
        # PASSO 6: Decodificação (se Manchester foi usado)
        if use_manchester:
            # Remove codificação Manchester para recuperar bits originais
            bits_recebidos = manchester_decode(bits_demodulados)
        else:
            bits_recebidos = bits_demodulados
        
        # PASSO 7: Calcular BER comparando bits recebidos com bits originais
        ber, numero_erros, total_bits = bit_error_rate(bits_originais, bits_recebidos)
        resultados_ber.append(ber)
        
        # Mostrar comparação para cálculo de BER
        amostra = min(16, len(bits_originais))
        orig_str = ''.join(str(b) for b in bits_originais[:amostra])
        rec_str = ''.join(str(b) for b in bits_recebidos[:amostra])
        if len(bits_originais) > amostra:
            orig_str += "..."
            rec_str += "..."
        print(f"  [7. CÁLCULO BER]")
        print(f"     Bits originais: [{orig_str}] (tamanho: {len(bits_originais)})")
        print(f"     Bits recebidos: [{rec_str}] (tamanho: {len(bits_recebidos)})")
        print(f"     Erros: {numero_erros} de {total_bits} bits → BER = {ber:.6f} ({ber*100:.2f}%)")
        
        # Tentar decodificar mensagem recebida (mesmo com erros)
        try:
            mensagem_recebida = bits_to_text(bits_recebidos)
            # Mostrar apenas primeiros caracteres se muito longa
            if len(mensagem_recebida) > 50:
                mensagem_display = mensagem_recebida[:50] + "..."
            else:
                mensagem_display = mensagem_recebida
        except:
            mensagem_display = "[Erro ao decodificar - muitos erros]"
        
        # Exibir resultado
        percentual_erro = ber * 100
        if numero_erros == 0:
            print(f"  [BPSK] SNR = {snr_eb_n0_db:4.1f} dB → {numero_erros} erros em {total_bits} bits ({percentual_erro:.2f}%) ✓ PERFEITO!")
            print(f"         Mensagem recebida: '{mensagem_display}'")
        else:
            print(f"  [BPSK] SNR = {snr_eb_n0_db:4.1f} dB → {numero_erros} erros em {total_bits} bits ({percentual_erro:.2f}%)")
            print(f"         Mensagem recebida: '{mensagem_display}'")
    
    return resultados_ber


def simulate_ber_qpsk(bits: np.ndarray, snr_db_values: List[float], use_manchester: bool = True, use_carrier: bool = False, fc: float = 1.0, fs: float = 10.0) -> List[float]:
    """
    Simula a Taxa de Erro de Bits (BER) para modulação QPSK.
    
    Fluxo do processo:
    1. Codificação (opcional): aplica Manchester encoding se solicitado
    2. Modulação: converte bits em símbolos QPSK (2 bits por símbolo)
    2.5. Portadora (opcional): adiciona portadora para modulação passa-banda
    3. Conversão SNR: transforma Eb/N0 (energia por bit) em Es/N0 (energia por símbolo)
    4. Canal: adiciona ruído AWGN ao sinal
    4.5. Remoção portadora (opcional): remove portadora se foi adicionada
    5. Demodulação: converte símbolos de volta para bits
    6. Decodificação (opcional): remove Manchester encoding se foi aplicado
    7. Cálculo BER: compara bits recebidos com bits originais
    
    Parâmetros:
        bits: bits de informação originais
        snr_db_values: lista de valores de SNR em dB (Eb/N0)
        use_manchester: se True, aplica codificação Manchester
        use_carrier: se True, adiciona portadora (modulação passa-banda)
        fc: frequência da portadora em Hz (padrão: 1.0)
        fs: taxa de amostragem (amostras por símbolo) (padrão: 10.0)
    
    Retorna:
        Lista de valores de BER para cada SNR
    """
    # PASSO 1: Preparar bits para transmissão
    bits_originais = bits.copy()
    bits_para_transmitir = bits_originais
    
    # Aplicar codificação Manchester se solicitado
    if use_manchester:
        bits_para_transmitir = manchester_encode(bits_originais)
        print(f"  [QPSK] Bits originais: {len(bits_originais)}, após Manchester: {len(bits_para_transmitir)}")

    # Lista para armazenar resultados de BER
    resultados_ber = []
    
    # Para cada valor de SNR, realizar simulação
    for snr_eb_n0_db in snr_db_values:
        print(f"\n  --- Simulando QPSK com SNR = {snr_eb_n0_db:.1f} dB ---")
        
        # PASSO 2: Modulação QPSK
        # Converte bits em símbolos QPSK (2 bits por símbolo)
        # Retorna também informação sobre padding se necessário
        simbolos_banda_base, padding_bits = qpsk_modulate(bits_para_transmitir)
        
        # PASSO 2.5: Adicionar portadora (opcional)
        # Em sistemas reais: fc >> taxa_de_símbolos (portadora muito maior)
        # Aplica pulse shaping para tornar mais realista
        if use_carrier:
            sinal_transmitido = add_carrier(simbolos_banda_base, fc, fs, use_pulse_shaping=True)
        else:
            sinal_transmitido = simbolos_banda_base
        
        # PASSO 3: Conversão de Eb/N0 para Es/N0
        # Eb/N0 = energia por bit de informação
        # Es/N0 = energia por símbolo modulado
        #
        # Com Manchester: cada bit de informação vira 2 bits, que vira 1 símbolo QPSK
        # Portanto: 1 bit info = 1 símbolo QPSK, então Es/N0 = Eb/N0
        #
        # Sem Manchester: 2 bits de informação vira 1 símbolo QPSK
        # Portanto: Es = 2*Eb, então Es/N0 = Eb/N0 + 10*log10(2) ≈ Eb/N0 + 3 dB
        if use_manchester:
            # 1 bit de informação = 1 símbolo QPSK, sem ajuste necessário
            snr_es_n0_db = snr_eb_n0_db
        else:
            # Sem Manchester: 2 bits = 1 símbolo, então a energia por símbolo é maior
            fator_qpsk_db = 10 * np.log10(2)  # ≈ 3.01 dB
            snr_es_n0_db = snr_eb_n0_db + fator_qpsk_db
        
        # PASSO 4: Simular canal com ruído AWGN
        # Adiciona ruído gaussiano branco ao sinal transmitido
        sinal_com_ruido = add_awgn(sinal_transmitido, snr_es_n0_db)
        
        # PASSO 4.5: Remover portadora (se foi adicionada)
        # Em sistemas reais, requer sincronização precisa (PLL - Phase Locked Loop)
        # Aplica filtro passa-baixa para remover componentes de alta frequência
        if use_carrier:
            simbolos_recebidos = remove_carrier(sinal_com_ruido, fc, fs, use_filtering=True)
        else:
            simbolos_recebidos = sinal_com_ruido
        
        # PASSO 5: Demodulação QPSK
        # Converte símbolos recebidos de volta para bits
        # Remove padding se foi adicionado durante modulação
        bits_demodulados = qpsk_demodulate(simbolos_recebidos, padding_bits)
        
        # PASSO 6: Decodificação (se Manchester foi usado)
        if use_manchester:
            # Remove codificação Manchester para recuperar bits originais
            bits_recebidos = manchester_decode(bits_demodulados)
        else:
            bits_recebidos = bits_demodulados
        
        # PASSO 7: Calcular BER comparando bits recebidos com bits originais
        ber, numero_erros, total_bits = bit_error_rate(bits_originais, bits_recebidos)
        resultados_ber.append(ber)
        
        # Mostrar comparação para cálculo de BER
        amostra = min(16, len(bits_originais))
        orig_str = ''.join(str(b) for b in bits_originais[:amostra])
        rec_str = ''.join(str(b) for b in bits_recebidos[:amostra])
        if len(bits_originais) > amostra:
            orig_str += "..."
            rec_str += "..."
        print(f"  [7. CÁLCULO BER]")
        print(f"     Bits originais: [{orig_str}] (tamanho: {len(bits_originais)})")
        print(f"     Bits recebidos: [{rec_str}] (tamanho: {len(bits_recebidos)})")
        print(f"     Erros: {numero_erros} de {total_bits} bits → BER = {ber:.6f} ({ber*100:.2f}%)")
        
        # Tentar decodificar mensagem recebida (mesmo com erros)
        try:
            mensagem_recebida = bits_to_text(bits_recebidos)
            # Mostrar apenas primeiros caracteres se muito longa
            if len(mensagem_recebida) > 50:
                mensagem_display = mensagem_recebida[:50] + "..."
            else:
                mensagem_display = mensagem_recebida
        except:
            mensagem_display = "[Erro ao decodificar - muitos erros]"
        
        # Exibir resultado
        percentual_erro = ber * 100
        if numero_erros == 0:
            print(f"  [QPSK] SNR = {snr_eb_n0_db:4.1f} dB → {numero_erros} erros em {total_bits} bits ({percentual_erro:.2f}%) ✓ PERFEITO!")
            print(f"         Mensagem recebida: '{mensagem_display}'")
        else:
            print(f"  [QPSK] SNR = {snr_eb_n0_db:4.1f} dB → {numero_erros} erros em {total_bits} bits ({percentual_erro:.2f}%)")
            print(f"         Mensagem recebida: '{mensagem_display}'")
    
    return resultados_ber


def run_full_simulation(
    message: str = "Trabalho de Comunicacao Digital",
    snr_db_values: List[float] | None = None,
    output_dir: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa uma simulação completa de comunicação digital.
    
    Processo:
    1. Converte mensagem de texto em bits
    2. Simula BER para BPSK e QPSK com diferentes valores de SNR
    3. Salva resultados em arquivo de texto
    4. Gera gráfico comparativo
    
    Parâmetros:
        message: mensagem de texto a ser transmitida
        snr_db_values: lista de valores de SNR em dB (Eb/N0) para testar
        output_dir: diretório onde salvar resultados (None = usa pasta 'output')
    
    Retorna:
        (array_SNR, array_BER_BPSK, array_BER_QPSK)
    """
    # Configurar valores padrão se não fornecidos
    if snr_db_values is None:
        snr_db_values = [0, 2, 4, 6, 8, 10]

    # Configurar diretório de saída
    if output_dir is None:
        diretorio_base = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(diretorio_base, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Cabeçalho da simulação
    print("=" * 60)
    print("Simulação de Comunicação Digital")
    print("=" * 60)
    print(f"Mensagem: '{message}'")
    
    # Converter mensagem de texto em bits
    bits_originais = text_to_bits(message)
    print(f"Bits originais: {len(bits_originais)} bits")
    print(f"Valores de SNR (Eb/N0): {snr_db_values} dB")
    print()
    
    # Explicar conceitos importantes
    print("BER = Bit Error Rate (Taxa de Erro de Bits)")
    print("     = número de bits errados / total de bits")
    print("     = 0.0 = sem erros, 1.0 = todos errados")
    print()
    print("SNR = Eb/N0 (Energia por bit de informação / Densidade espectral de ruído)")
    print("     O código converte automaticamente para Es/N0 (energia por símbolo)")
    print("     antes de adicionar ruído ao canal")
    print()

    # FEATURE FLAG: Análise com portadora (modulação passa-banda)
    # Por padrão: DESATIVADO - não é essencial para demonstrar os conceitos principais
    # Para ativar: mudar ENABLE_CARRIER_ANALYSIS = True
    ENABLE_CARRIER_ANALYSIS = False
    
    # Simular BPSK
    print("Simulando BPSK...")
    resultados_ber_bpsk = simulate_ber_bpsk(bits_originais, snr_db_values, use_manchester=True, use_carrier=False)
    print()
    
    # Simular QPSK
    print("Simulando QPSK...")
    resultados_ber_qpsk = simulate_ber_qpsk(bits_originais, snr_db_values, use_manchester=True, use_carrier=False)
    print()
    
    # Análise com portadora (OPCIONAL - apenas se ENABLE_CARRIER_ANALYSIS = True)
    if ENABLE_CARRIER_ANALYSIS:
        print("=" * 60)
        print("ANÁLISE OPCIONAL: Impacto da Portadora (Modulação Passa-Banda)")
        print("=" * 60)
        print("Esta análise demonstra o impacto de adicionar portadora ao sinal.")
        print("Não é essencial para os objetivos principais do trabalho.\n")
        
        # Simular BPSK com portadora
        print("Simulando BPSK (com portadora fc=1.0 Hz, fs=10.0)...")
        resultados_ber_bpsk_com = simulate_ber_bpsk(bits_originais, snr_db_values, use_manchester=True, use_carrier=True, fc=1.0, fs=10.0)
        print()
        
        # Simular QPSK com portadora
        print("Simulando QPSK (com portadora fc=1.0 Hz, fs=10.0)...")
        resultados_ber_qpsk_com = simulate_ber_qpsk(bits_originais, snr_db_values, use_manchester=True, use_carrier=True, fc=1.0, fs=10.0)
        print()
        
        # Armazenar resultados com portadora para análise comparativa
        resultados_ber_bpsk_com_array = np.array(resultados_ber_bpsk_com, dtype=float)
        resultados_ber_qpsk_com_array = np.array(resultados_ber_qpsk_com, dtype=float)
    else:
        # Se não usar portadora, criar arrays vazios para compatibilidade
        resultados_ber_bpsk_com_array = np.array([])
        resultados_ber_qpsk_com_array = np.array([])

    # Converter resultados para arrays numpy para facilitar manipulação
    array_snr = np.array(snr_db_values, dtype=float)
    array_ber_bpsk = np.array(resultados_ber_bpsk, dtype=float)
    array_ber_qpsk = np.array(resultados_ber_qpsk, dtype=float)
    
    # Análise comparativa: impacto da portadora (apenas se ativado)
    if ENABLE_CARRIER_ANALYSIS and len(resultados_ber_bpsk_com_array) > 0:
        print("=" * 60)
        print("ANÁLISE OPCIONAL: Impacto da Portadora")
        print("=" * 60)
        print("\nMelhorias implementadas para aproximar da realidade:")
        print("✓ Pulse shaping (formatação de pulso) - reduz ISI")
        print("✓ Filtro passa-baixa na demodulação - remove componentes 2fc")
        print("✓ Downsampling adequado após filtragem")
        print("\nDiferenças principais:")
        print("1. BPSK: usa apenas cos(2πfc*t) - mais simples, menos eficiente")
        print("2. QPSK: usa cos(2πfc*t) e sin(2πfc*t) - mais eficiente, mais complexo")
        print("\nComparação de BER (com vs sem portadora):")
        print(f"{'SNR (dB)':<10} {'BPSK sem':<12} {'BPSK com':<12} {'QPSK sem':<12} {'QPSK com':<12}")
        print("-" * 60)
        for i, snr in enumerate(array_snr):
            bpsk_sem = array_ber_bpsk[i]
            bpsk_com = resultados_ber_bpsk_com_array[i]
            qpsk_sem = array_ber_qpsk[i]
            qpsk_com = resultados_ber_qpsk_com_array[i]
            print(f"{snr:>6.1f}     {bpsk_sem:>10.6f}  {bpsk_com:>10.6f}  {qpsk_sem:>10.6f}  {qpsk_com:>10.6f}")
        print("=" * 60)
        print()

    # Salvar resultados em arquivo de texto
    caminho_arquivo_texto = os.path.join(output_dir, "ber_results_bpsk_qpsk.txt")
    with open(caminho_arquivo_texto, "w", encoding="utf-8") as arquivo:
        # Cabeçalho do arquivo
        if ENABLE_CARRIER_ANALYSIS and len(resultados_ber_bpsk_com_array) > 0:
            arquivo.write("SNR (dB)\tBER_BPSK_sem\tBER_BPSK_com\tBER_QPSK_sem\tBER_QPSK_com\tBER_BPSK_sem (%)\tBER_BPSK_com (%)\tBER_QPSK_sem (%)\tBER_QPSK_com (%)\n")
            # Escrever cada linha de resultados
            for snr, bpsk_sem, bpsk_com, qpsk_sem, qpsk_com in zip(array_snr, array_ber_bpsk, resultados_ber_bpsk_com_array, array_ber_qpsk, resultados_ber_qpsk_com_array):
                arquivo.write(f"{snr:.1f}\t\t{bpsk_sem:.6f}\t\t{bpsk_com:.6f}\t\t{qpsk_sem:.6f}\t\t{qpsk_com:.6f}\t\t{bpsk_sem*100:.2f}%\t\t{bpsk_com*100:.2f}%\t\t{qpsk_sem*100:.2f}%\t\t{qpsk_com*100:.2f}%\n")
        else:
            arquivo.write("SNR (dB)\tBER_BPSK\t\tBER_QPSK\t\tBER_BPSK (%)\tBER_QPSK (%)\n")
            arquivo.write("-" * 60 + "\n")
            # Escrever cada linha de resultados
            for snr, ber_bpsk, ber_qpsk in zip(array_snr, array_ber_bpsk, array_ber_qpsk):
                # Formato mais legível: decimal e porcentagem
                arquivo.write(f"{snr:.1f}\t\t{ber_bpsk:.6f}\t\t{ber_qpsk:.6f}\t\t{ber_bpsk*100:.2f}%\t\t{ber_qpsk*100:.2f}%\n")
    print(f"Resultados salvos em: {caminho_arquivo_texto}")

    # Gerar gráfico comparativo
    caminho_grafico = os.path.join(output_dir, "ber_curve_bpsk_qpsk.png")
    
    # Criar figura
    plt.figure()
    
    # Plotar curvas BER x SNR em escala logarítmica
    plt.semilogy(array_snr, array_ber_bpsk, marker="o", label="BPSK", linewidth=2, linestyle="-")
    plt.semilogy(array_snr, array_ber_qpsk, marker="s", label="QPSK", linewidth=2, linestyle="-")
    
    # Adicionar curvas com portadora apenas se análise estiver ativada
    if ENABLE_CARRIER_ANALYSIS and len(resultados_ber_bpsk_com_array) > 0:
        plt.semilogy(array_snr, resultados_ber_bpsk_com_array, marker="o", label="BPSK (com portadora)", linewidth=2, linestyle="--", alpha=0.7)
        plt.semilogy(array_snr, resultados_ber_qpsk_com_array, marker="s", label="QPSK (com portadora)", linewidth=2, linestyle="--", alpha=0.7)
    
    # Configurar eixos e título
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("BER", fontsize=12)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=11)
    if ENABLE_CARRIER_ANALYSIS:
        plt.title("Curva BER x SNR – BPSK vs QPSK (com/sem portadora)", fontsize=13)
    else:
        plt.title("Curva BER x SNR – BPSK vs QPSK (com Manchester)", fontsize=13)
    
    # Salvar gráfico
    plt.savefig(caminho_grafico, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {caminho_grafico}")
    
    # Finalização
    print("=" * 60)
    print("Simulação concluída!")
    print("=" * 60)

    return array_snr, array_ber_bpsk, array_ber_qpsk
