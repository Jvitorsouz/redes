import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# ==============================================================================
# PARÂMETROS GLOBAIS DA SIMULAÇÃO
# ==============================================================================
SAMPLE_RATE = 44100
BIT_DURATION = 0.1  # Reduzido para uma simulação mais rápida
FREQ_LOW = 440
FREQ_HIGH = 880
THRESHOLD = (FREQ_LOW + FREQ_HIGH) / 2

# ==============================================================================
# FUNÇÕES DE CODIFICAÇÃO E GERAÇÃO DE TOM
# ==============================================================================

def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE):
    """Gera um tom senoidal com janela de Hanning."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    window = np.hanning(len(tone))
    return tone * window

def encode_nrz(data_bits):
    """Codifica dados usando NRZ."""
    audio_signal = np.array([])
    for bit in data_bits:
        freq = FREQ_HIGH if bit == '1' else FREQ_LOW
        tone = generate_tone(freq, BIT_DURATION)
        audio_signal = np.concatenate([audio_signal, tone])
    return audio_signal

def encode_manchester(data_bits):
    """Codifica dados usando Manchester."""
    audio_signal = np.array([])
    for bit in data_bits:
        if bit == '1': # Alto -> Baixo
            tone1 = generate_tone(FREQ_HIGH, BIT_DURATION / 2)
            tone2 = generate_tone(FREQ_LOW, BIT_DURATION / 2)
        else: # Baixo -> Alto
            tone1 = generate_tone(FREQ_LOW, BIT_DURATION / 2)
            tone2 = generate_tone(FREQ_HIGH, BIT_DURATION / 2)
        bit_signal = np.concatenate([tone1, tone2])
        audio_signal = np.concatenate([audio_signal, bit_signal])
    return audio_signal

# ==============================================================================
# FUNÇÃO PARA ADICIONAR RUÍDO
# ==============================================================================

def adicionar_ruido(audio_signal, snr_db):
    """Adiciona ruído gaussiano a um sinal de áudio."""
    signal_power = np.mean(audio_signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_signal))
    return audio_signal + noise

# ==============================================================================
# FUNÇÕES DE DECODIFICAÇÃO
# ==============================================================================

def detect_frequency(audio_segment, sample_rate=SAMPLE_RATE):
    """Detecta a frequência dominante em um segmento de áudio."""
    # FIX: Adicionada verificação para garantir que o segmento não seja muito pequeno
    if len(audio_segment) < 2:
        return 0
    fft_spectrum = np.fft.fft(audio_segment)
    freqs = np.fft.fftfreq(len(fft_spectrum), 1 / sample_rate)
    magnitude = np.abs(fft_spectrum[:len(fft_spectrum)//2])
    peak_idx = np.argmax(magnitude)
    return abs(freqs[peak_idx])

def frequency_to_bit(frequency, threshold=THRESHOLD):
    """Converte uma frequência em um bit com base em um limiar."""
    return '1' if frequency > threshold else '0'

def decode_nrz(audio_signal, num_bits, sample_rate=SAMPLE_RATE):
    """Decodifica um sinal NRZ."""
    samples_per_bit = int(sample_rate * BIT_DURATION)
    decoded_bits = ""
    for i in range(num_bits):
        segment = audio_signal[i*samples_per_bit : (i+1)*samples_per_bit]
        # Analisa o meio do bit para evitar transições
        mid_segment = segment[len(segment)//4 : -len(segment)//4]
        freq = detect_frequency(mid_segment)
        decoded_bits += frequency_to_bit(freq)
    return decoded_bits

def decode_manchester(audio_signal, num_bits, sample_rate=SAMPLE_RATE):
    """Decodifica um sinal Manchester."""
    samples_per_bit = int(sample_rate * BIT_DURATION)
    decoded_bits = ""
    for i in range(num_bits):
        bit_segment = audio_signal[i*samples_per_bit : (i+1)*samples_per_bit]
        mid_point = len(bit_segment) // 2
        
        # FIX: Lógica de fatiamento corrigida para pegar uma porção significativa de cada metade
        margin = mid_point // 4 # Pega o meio de cada metade, descartando as bordas
        first_half = bit_segment[margin : mid_point - margin]
        second_half = bit_segment[mid_point + margin : -margin]
        
        freq1 = detect_frequency(first_half)
        state1 = frequency_to_bit(freq1)
        
        freq2 = detect_frequency(second_half)
        state2 = frequency_to_bit(freq2)
        
        if state1 == '1' and state2 == '0':
            decoded_bits += '1'
        elif state1 == '0' and state2 == '1':
            decoded_bits += '0'
        else:
            decoded_bits += '?' # Erro de decodificação
    return decoded_bits

# ==============================================================================
# FUNÇÃO PRINCIPAL DO EXPERIMENTO
# ==============================================================================

def run_snr_experiment(modulation_type, original_bits, snr_range):
    """Executa a simulação para uma modulação em uma faixa de SNRs."""
    error_counts = []
    
    if modulation_type == 'nrz':
        encode_func = encode_nrz
        decode_func = decode_nrz
    elif modulation_type == 'manchester':
        encode_func = encode_manchester
        decode_func = decode_manchester
    else:
        raise ValueError("Modulação desconhecida")

    # 1. Gera o sinal limpo uma única vez
    clean_signal = encode_func(original_bits)
    
    # 2. Itera sobre cada nível de SNR
    for snr in snr_range:
        # Adiciona ruído
        noisy_signal = adicionar_ruido(clean_signal, snr)
        
        # Decodifica
        decoded_bits = decode_func(noisy_signal, len(original_bits))
        
        # Conta os erros
        errors = sum(1 for orig, dec in zip(original_bits, decoded_bits) if orig != dec)
        error_counts.append(errors)
        
    return error_counts

def analyze_results(snr_range, error_counts, modulation_name):
    """Analisa e imprime os pontos críticos de falha."""
    first_error_snr = None
    total_fail_snr = None
    
    for snr, errors in zip(snr_range, error_counts):
        if errors > 0 and first_error_snr is None:
            first_error_snr = snr
        if errors >= len(original_bits) / 2 and total_fail_snr is None:
            total_fail_snr = snr
            
    print(f"\n--- Análise para {modulation_name.upper()} ---")
    if first_error_snr is not None:
        print(f"a) Primeiro erro de bit ocorreu em SNR = {first_error_snr} dB")
    else:
        print("a) Nenhum erro detectado na faixa de SNR testada.")
        
    if total_fail_snr is not None:
        print(f"b) Falha significativa (>=50% de erro) começou em SNR = {total_fail_snr} dB")
    else:
        print("b) Nenhuma falha significativa detectada.")


# ==============================================================================
# BLOCO DE EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    original_bits = "00111000"
    # Faixa de SNR para testar: de 10dB até -20dB
    snr_range = np.arange(10, -21, -1)

    print("Iniciando simulação de robustez de modulação...")
    print(f"Mensagem original: {original_bits}")

    # Executa os experimentos
    nrz_errors = run_snr_experiment('nrz', original_bits, snr_range)
    manchester_errors = run_snr_experiment('manchester', original_bits, snr_range)

    # Analisa e imprime os resultados textuais
    analyze_results(snr_range, nrz_errors, 'NRZ')
    analyze_results(snr_range, manchester_errors, 'Manchester')

    # Plota o gráfico
    plt.figure(figsize=(12, 7))
    plt.plot(snr_range, nrz_errors, 'o-', label='NRZ', color='royalblue')
    plt.plot(snr_range, manchester_errors, 's-', label='Manchester', color='darkorange')
    
    # Configurações do gráfico
    plt.title('Comparação de Robustez: NRZ vs. Manchester')
    plt.xlabel('Relação Sinal-Ruído (SNR) em dB')
    plt.ylabel('Número de Bits Errados')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Inverte o eixo X para que a performance degrade da direita para a esquerda
    plt.gca().invert_xaxis()
    
    # Garante que o eixo Y tenha apenas números inteiros
    plt.yticks(np.arange(0, len(original_bits) + 1, 1))
    
    plt.show()
