import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PARÂMETROS DE DECODIFICAÇÃO
# ATENÇÃO: Estes valores DEVEM ser os mesmos usados para criar o áudio!
# ==============================================================================
SAMPLE_RATE = 44100   # Taxa de amostragem do audio em Hz
BIT_DURATION = 1.0    # Duração de cada bit em segundos (ex: 1.0, 0.5, 0.1)
FREQ_LOW = 440        # Frequência que representa o bit '0'
FREQ_HIGH = 880       # Frequência que representa o bit '1'
# Limiar para decisão. Geralmente, o ponto médio entre FREQ_LOW e FREQ_HIGH.
THRESHOLD = (FREQ_LOW + FREQ_HIGH) / 2

# ==============================================================================
# FUNÇÕES DE DECODIFICAÇÃO (Mantidas do seu código original)
# ==============================================================================

def show(data:str, debug):
    """Função auxiliar para imprimir mensagens de debug."""
    if debug:
        print(data)

def detect_frequency(audio_segment, sample_rate):
    """Detecta a frequência dominante em um segmento de áudio usando FFT."""
    # FFT para análise espectral
    fft_spectrum = np.fft.fft(audio_segment)
    freqs = np.fft.fftfreq(len(fft_spectrum), 1 / sample_rate)

    # Considera apenas frequências positivas
    magnitude = np.abs(fft_spectrum[:len(fft_spectrum)//2])
    freqs_positive = freqs[:len(freqs)//2]

    # Encontra o pico de frequência
    try:
        peak_idx = np.argmax(magnitude)
        detected_freq = abs(freqs_positive[peak_idx])
    except IndexError:
        # Retorna 0 se o segmento de áudio for vazio
        return 0.0
        
    return detected_freq

def frequency_to_bit(frequency, threshold=THRESHOLD):
    """Converte frequência detectada em bit com base no limiar."""
    return '1' if frequency > threshold else '0'

def decode_nrz(audio_signal, num_bits, sample_rate, debug=False):
    """Decodifica um sinal de áudio usando a lógica NRZ."""
    samples_per_bit = int(sample_rate * BIT_DURATION)
    decoded_bits = ""
    show("Decodificando NRZ...", debug)

    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit

        if end_idx > len(audio_signal):
            show(f"Aviso: Áudio muito curto para {num_bits} bits. Parando no bit {i}.", debug)
            break

        # Analisa o meio do bit para evitar problemas nas transições
        mid_start = start_idx + samples_per_bit // 4
        mid_end = end_idx - samples_per_bit // 4
        segment = audio_signal[mid_start:mid_end]

        freq = detect_frequency(segment, sample_rate)
        bit = frequency_to_bit(freq)
        decoded_bits += bit
        show(f"Bit {i}: freq={freq:.1f}Hz -> '{bit}'", debug)

    return decoded_bits

def decode_manchester(audio_signal, num_bits, sample_rate, debug=False):
    """Decodifica um sinal de áudio usando a lógica Manchester."""
    samples_per_bit = int(sample_rate * BIT_DURATION)
    decoded_bits = ""
    show("Decodificando Manchester...", debug)

    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit

        if end_idx > len(audio_signal):
            show(f"Aviso: Áudio muito curto para {num_bits} bits. Parando no bit {i}.", debug)
            break

        # Analisa a primeira e a segunda metade do bit
        mid_point = start_idx + samples_per_bit // 2
        
        # Define os segmentos para análise (com uma pequena margem para evitar transições)
        first_half = audio_signal[start_idx + samples_per_bit//8 : mid_point - samples_per_bit//8]
        second_half = audio_signal[mid_point + samples_per_bit//8 : end_idx - samples_per_bit//8]
        
        freq1 = detect_frequency(first_half, sample_rate)
        state1 = frequency_to_bit(freq1)
        
        freq2 = detect_frequency(second_half, sample_rate)
        state2 = frequency_to_bit(freq2)

        # Determina o bit baseado na transição de frequência
        if state1 == '1' and state2 == '0':  # Alto -> Baixo
            bit = '1'
            show(f"Bit {i}: {freq1:.1f}Hz -> {freq2:.1f}Hz = alto->baixo = '1'", debug)
        elif state1 == '0' and state2 == '1':  # Baixo -> Alto
            bit = '0'
            show(f"Bit {i}: {freq1:.1f}Hz -> {freq2:.1f}Hz = baixo->alto = '0'", debug)
        else:  # Erro ou transição inválida
            bit = '?'
            show(f"Bit {i}: {freq1:.1f}Hz -> {freq2:.1f}Hz = ERRO na transição", debug)
        
        decoded_bits += bit

    return decoded_bits

def plot_signal(audio_signal, sample_rate, title, num_bits):
    """Plota o sinal de áudio e as divisões dos bits."""
    time_axis = np.linspace(0, len(audio_signal) / sample_rate, num=len(audio_signal))
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, audio_signal)
    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Desenha linhas verticais para marcar onde cada bit deveria terminar
    for i in range(1, num_bits):
        plt.axvline(x=i * BIT_DURATION, color='red', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.show()

# ==============================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# ==============================================================================

def decodificar_arquivo(nome_arquivo, tipo_codificacao, mostrar_grafico=True, modo_debug=False):
    """
    Função principal que carrega um arquivo de áudio e o decodifica.
    """
    print(f"--- Iniciando decodificação do arquivo: '{nome_arquivo}' ---")
    print(f"--- Usando o método: {tipo_codificacao.upper()} ---")

    try:
        # 1. Carrega o arquivo de áudio
        audio_data, taxa_amostra_arquivo = sf.read(nome_arquivo, dtype='float32')

        # Se o áudio for estéreo, pega apenas um canal (o esquerdo)
        if audio_data.ndim > 1:
            print("Áudio estéreo detectado. Usando apenas o canal esquerdo.")
            audio_data = audio_data[:, 0]

        # 2. Calcula o número de bits com base na duração do áudio
        duracao_total = len(audio_data) / taxa_amostra_arquivo
        numero_de_bits = int(round(duracao_total / BIT_DURATION))

        print("\nInformações do áudio:")
        print(f"\tDuração total: {duracao_total:.2f} segundos")
        print(f"\tTaxa de amostragem: {taxa_amostra_arquivo} Hz")
        print(f"\tNúmero de bits estimado: {numero_de_bits}")

        if numero_de_bits == 0:
            print("\nERRO: Áudio muito curto para decodificar.")
            return

        # 3. Chama a função de decodificação correta
        bits_decodificados = ""
        if tipo_codificacao.lower() == 'nrz':
            bits_decodificados = decode_nrz(audio_data, numero_de_bits, taxa_amostra_arquivo, debug=modo_debug)
        elif tipo_codificacao.lower() == 'manchester':
            bits_decodificados = decode_manchester(audio_data, numero_de_bits, taxa_amostra_arquivo, debug=modo_debug)
        else:
            print("\nERRO: Tipo de codificação desconhecido. Use 'nrz' ou 'manchester'.")
            return

        print("\n" + "="*50)
        print(f"  BITS DECODIFICADOS: {bits_decodificados}")
        print("="*50 + "\n")

        # 4. Opcionalmente, plota o gráfico do sinal
        if mostrar_grafico:
            plot_signal(audio_data, taxa_amostra_arquivo, f"Sinal de '{nome_arquivo}' - Decodificação {tipo_codificacao.upper()}", numero_de_bits)

    except FileNotFoundError:
        print(f"\nERRO CRÍTICO: Arquivo '{nome_arquivo}' não encontrado!")
        print("Por favor, verifique se o nome está correto e se o arquivo está na mesma pasta que este script.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante o processamento: {e}")

# --- CONFIGURAÇÕES ---
# Altere as duas linhas abaixo para corresponder ao seu arquivo
NOME_DO_ARQUIVO = "dados_22_44100hz.wav"  # <-- COLOQUE O NOME DO SEU ARQUIVO .WAV AQUI
TIPO_DA_CODIFICACAO = "nrz"      # <-- DIGITE 'nrz' ou 'manchester'

# --- EXECUÇÃO ---
# Chama a função para decodificar o seu arquivo
decodificar_arquivo(NOME_DO_ARQUIVO, "nrz", mostrar_grafico=True, modo_debug=False)
