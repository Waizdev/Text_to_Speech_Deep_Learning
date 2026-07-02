class Config:

    # ==========================
    # Dataset
    # ==========================

    SAMPLE_RATE = 22050

    N_MELS = 80

    N_FFT = 1024

    HOP_LENGTH = 256

    WIN_LENGTH = 1024

    FMIN = 0

    FMAX = 8000

    # ==========================
    # Text
    # ==========================

    VOCAB_SIZE = 80

    MAX_TEXT_LENGTH = 200

    # ==========================
    # Model
    # ==========================

    EMBEDDING_DIM = 256

    ENCODER_DIM = 256

    ATTENTION_DIM = 128

    ATTENTION_RNN_DIM = 256

    DECODER_DIM = 256

    PRENET_DIM = 256

    POSTNET_DIM = 512

    # ==========================
    # Training
    # ==========================

    BATCH_SIZE = 32

    LEARNING_RATE = 1e-3

    EPOCHS = 100

    TEACHER_FORCING_RATIO = 1.0

    MAX_DECODER_STEPS = 1000
