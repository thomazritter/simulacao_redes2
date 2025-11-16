
import numpy as np


def text_to_bits(text: str) -> np.ndarray:
    """
    Converte uma string de texto em um vetor de bits.
    
    Processo:
    1. Para cada caractere na string, obtém seu código ASCII
    2. Converte o código ASCII para binário (8 bits)
    3. Adiciona cada bit ao vetor final
    
    Exemplo:
        'A' (ASCII 65) -> '01000001' -> [0, 1, 0, 0, 0, 0, 0, 1]
    
    Parâmetros:
        text: string de texto a ser convertida
    
    Retorna:
        Array numpy com bits (0 ou 1) representando o texto
    """
    if not isinstance(text, str):
        raise TypeError("text deve ser uma string")

    lista_bits = []
    
    # Para cada caractere na string
    for caractere in text:
        # Obter código ASCII do caractere
        codigo_ascii = ord(caractere)
        
        # Converter código ASCII para binário com 8 dígitos (completar com zeros à esquerda)
        # Exemplo: 65 -> '01000001'
        binario_8_bits = format(codigo_ascii, "08b")
        
        # Adicionar cada bit (como inteiro) à lista
        for bit_char in binario_8_bits:
            lista_bits.append(int(bit_char))
    
    # Converter lista para array numpy
    bits_array = np.array(lista_bits, dtype=np.uint8)
    
    # Mostrar conversão texto → bits
    amostra = min(32, len(bits_array))
    bits_str = ''.join(str(b) for b in bits_array[:amostra])
    if len(bits_array) > amostra:
        bits_str += "..."
    print(f"  [1. TEXTO → BITS]")
    print(f"     Texto: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"     Bits:  [{bits_str}] (tamanho: {len(bits_array)} bits)")
    
    return bits_array


def bits_to_text(bits: np.ndarray) -> str:
    """
    Converte um vetor de bits em uma string de texto.
    
    Processo:
    1. Agrupa bits em grupos de 8 (bytes)
    2. Para cada byte, reconstrói o valor decimal
    3. Converte o valor decimal em caractere ASCII
    4. Junta todos os caracteres em uma string
    
    Exemplo:
        [0, 1, 0, 0, 0, 0, 0, 1] -> 65 -> 'A'
    
    Parâmetros:
        bits: array numpy com bits (0 ou 1)
    
    Retorna:
        String de texto reconstruída
    """
    # Garantir que bits seja um array numpy
    bits_array = np.asarray(bits, dtype=np.uint8)
    
    # Validar formato
    if bits_array.ndim != 1:
        raise ValueError("bits deve ser um vetor 1D")
    if len(bits_array) % 8 != 0:
        raise ValueError("tamanho de bits deve ser múltiplo de 8 (cada caractere precisa de 8 bits)")

    lista_caracteres = []
    
    # Processar bits em grupos de 8 (cada grupo é um byte = um caractere)
    for indice_inicio in range(0, len(bits_array), 8):
        # Extrair 8 bits consecutivos (um byte)
        byte_bits = bits_array[indice_inicio:indice_inicio+8]
        
        # Reconstruir valor decimal do byte
        # Exemplo: [0,1,0,0,0,0,0,1] -> 0*128 + 1*64 + 0*32 + ... = 65
        valor_decimal = 0
        for bit in byte_bits:
            # Deslocar bits à esquerda e adicionar novo bit
            valor_decimal = (valor_decimal << 1) | int(bit)
        
        # Converter valor decimal em caractere ASCII
        caractere = chr(valor_decimal)
        lista_caracteres.append(caractere)
    
    # Juntar todos os caracteres em uma string
    texto_reconstruido = "".join(lista_caracteres)
    return texto_reconstruido


def manchester_encode(bits: np.ndarray) -> np.ndarray:
    """
    Codifica bits usando codificação Manchester.
    
    Codificação Manchester:
    - Cada bit é representado por uma transição no meio do período
    - Bit 0 -> transição de 1 para 0: (1, 0)
    - Bit 1 -> transição de 0 para 1: (0, 1)
    
    Vantagens:
    - Sempre há transição no meio do bit (facilita sincronização)
    - Não há componente DC (útil para transmissão)
    
    Desvantagem:
    - Dobra a taxa de transmissão (1 bit vira 2 bits)
    
    Parâmetros:
        bits: array numpy com bits a codificar (0 ou 1)
    
    Retorna:
        Array numpy com bits codificados (dobro do tamanho)
    """
    # Garantir que bits seja um array numpy
    bits_entrada = np.asarray(bits, dtype=np.uint8)
    
    # Validar formato
    if bits_entrada.ndim != 1:
        raise ValueError("bits deve ser um vetor 1D")

    # Criar array para bits codificados (dobro do tamanho)
    tamanho_original = len(bits_entrada)
    bits_codificados = np.empty(2 * tamanho_original, dtype=np.uint8)
    
    # Aplicar codificação Manchester:
    # Bit 0 -> (1, 0): primeiro bit = 1, segundo bit = 0
    # Bit 1 -> (0, 1): primeiro bit = 0, segundo bit = 1
    #
    # Posições pares (0, 2, 4, ...): primeiro bit de cada par
    bits_codificados[0::2] = 1 - bits_entrada  # Inverter: 0->1, 1->0
    
    # Posições ímpares (1, 3, 5, ...): segundo bit de cada par
    bits_codificados[1::2] = bits_entrada  # Manter: 0->0, 1->1
    
    # Mostrar conversão bits → Manchester
    amostra_antes = min(16, tamanho_original)
    amostra_depois = min(32, len(bits_codificados))
    bits_antes = ''.join(str(b) for b in bits_entrada[:amostra_antes])
    bits_depois = ''.join(str(b) for b in bits_codificados[:amostra_depois])
    if tamanho_original > amostra_antes:
        bits_antes += "..."
    if len(bits_codificados) > amostra_depois:
        bits_depois += "..."
    print(f"  [2. BITS → MANCHESTER]")
    print(f"     Antes:  [{bits_antes}] (tamanho: {tamanho_original})")
    print(f"     Depois: [{bits_depois}] (tamanho: {len(bits_codificados)})")
    
    return bits_codificados


def manchester_decode(encoded: np.ndarray) -> np.ndarray:
    """
    Decodifica bits codificados com Manchester.
    
    Processo:
    1. Agrupa bits em pares
    2. Para pares válidos: (1,0) -> 0, (0,1) -> 1
    3. Para pares inválidos (após ruído): usa o segundo bit como decisão
    
    Tratamento de erros:
    - Pares válidos: (1,0) ou (0,1) -> decodificação correta
    - Pares inválidos após ruído: (0,0) ou (1,1)
      -> usa segundo bit: (0,0) -> 0, (1,1) -> 1
    
    Parâmetros:
        encoded: array numpy com bits codificados (deve ter tamanho par)
    
    Retorna:
        Array numpy com bits decodificados (metade do tamanho)
    """
    # Garantir que encoded seja um array numpy
    bits_codificados = np.asarray(encoded, dtype=np.uint8)
    
    # Validar formato
    if bits_codificados.ndim != 1:
        raise ValueError("encoded deve ser um vetor 1D")
    if len(bits_codificados) % 2 != 0:
        raise ValueError("tamanho de encoded deve ser par (cada bit codificado tem 2 bits)")

    # Extrair primeiro e segundo bit de cada par
    # Posições pares: primeiro bit de cada par
    primeiro_bit_cada_par = bits_codificados[0::2]
    # Posições ímpares: segundo bit de cada par
    segundo_bit_cada_par = bits_codificados[1::2]

    # Decodificação:
    # Para pares válidos: (1,0) -> 0, (0,1) -> 1
    # Para pares inválidos após ruído: (0,0) -> 0, (1,1) -> 1
    # 
    # Em todos os casos, o segundo bit representa o bit original:
    # - Par (1,0): segundo bit = 0 -> bit original = 0
    # - Par (0,1): segundo bit = 1 -> bit original = 1
    # - Par (0,0): segundo bit = 0 -> bit original = 0 (decisão após ruído)
    # - Par (1,1): segundo bit = 1 -> bit original = 1 (decisão após ruído)
    bits_decodificados = segundo_bit_cada_par.copy()
    
    # Mostrar conversão Manchester → bits
    amostra_antes = min(32, len(bits_codificados))
    amostra_depois = min(16, len(bits_decodificados))
    bits_antes = ''.join(str(b) for b in bits_codificados[:amostra_antes])
    bits_depois = ''.join(str(b) for b in bits_decodificados[:amostra_depois])
    if len(bits_codificados) > amostra_antes:
        bits_antes += "..."
    if len(bits_decodificados) > amostra_depois:
        bits_depois += "..."
    print(f"  [6. MANCHESTER → BITS]")
    print(f"     Antes:  [{bits_antes}] (tamanho: {len(bits_codificados)})")
    print(f"     Depois: [{bits_depois}] (tamanho: {len(bits_decodificados)})")
    
    return bits_decodificados
