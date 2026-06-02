# Day 061: Speech-to-Text Pipeline

## What You're Building

A complete automatic speech recognition (ASR) pipeline from raw audio waveforms to text transcriptions. You'll implement the core components that power systems like Whisper, DeepSpeech, and production ASR engines: audio feature extraction (Mel spectrograms), a sequence-to-sequence neural network with CTC (Connectionist Temporal Classification) loss, and a greedy decoder that converts model outputs back into readable text.

**Why it matters:** Speech recognition is one of the most commercially impactful AI applications — powering voice assistants, real-time captioning, call center analytics, and accessibility tools. Understanding the pipeline end-to-end (not just calling an API) lets you debug failures, optimize latency, and build custom models for domain-specific vocabularies.

## Core Concepts

### 1. Audio as a Signal

Sound is a pressure wave. When digitized, it becomes a sequence of amplitude samples at a fixed rate (e.g., 16,000 samples/second = 16kHz). A 5-second clip at 16kHz is 80,000 floating-point numbers. Feeding raw samples directly into a model is wasteful — most of the information is redundant or irrelevant to speech.

### 2. Mel Spectrograms — The Right Representation

The key insight: human hearing is logarithmic in frequency. We perceive the difference between 100Hz and 200Hz as much larger than between 5000Hz and 5100Hz. The Mel scale captures this:

```
mel(f) = 2595 * log10(1 + f / 700)
```

To build a Mel spectrogram:
1. **Window the signal**: Split the audio into overlapping frames (typically 25ms windows with 10ms hops). Apply a Hamming window to reduce spectral leakage.
2. **FFT each frame**: Convert from time domain to frequency domain using the Fast Fourier Transform. This gives you the power at each frequency for that time slice.
3. **Apply Mel filterbank**: Create triangular filters spaced according to the Mel scale. Multiply the power spectrum by each filter and sum — this compresses the frequency axis to match human perception.
4. **Log compression**: Take log(mel_energies + epsilon). This compresses the dynamic range, matching how we perceive loudness.

The result is a 2D matrix: (num_frames x num_mel_bins) — essentially an "image" of the audio that captures what matters for speech.

### 3. CTC — Solving the Alignment Problem

The fundamental challenge in ASR: we don't know which part of the audio corresponds to which character. CTC solves this elegantly:

- The model outputs a probability distribution over characters (plus a special "blank" token) at every time step.
- CTC considers ALL possible alignments between the output sequence and the target text.
- It introduces a "blank" token (ε) that the model can emit when it's between characters or uncertain.
- The CTC loss marginalizes over all valid alignments, so we never need explicit alignment labels.

**Collapsing rule**: To decode, merge consecutive duplicate characters and remove blanks.
- `h_h-ee-l-l-ll-oo` → `hello` (where `-` is blank)
- `aaa-bb` → `ab`

The math: CTC loss = -log P(target | input), where P is summed over all valid alignments using dynamic programming (forward-backward algorithm).

### 4. Greedy CTC Decoding

The simplest decoder: at each time step, pick the character with the highest probability, then apply the CTC collapsing rule. Fast but suboptimal — beam search with a language model does better in production, but greedy decoding captures the core idea.

### 5. Architecture: Conv + RNN + Linear

A typical CTC-based ASR model:
1. **Convolutional front-end**: 1-2 conv layers that downsample the spectrogram in time and extract local patterns (phoneme-like features).
2. **Recurrent layers**: Bidirectional GRUs/LSTMs that model temporal dependencies across the utterance.
3. **Linear projection**: Maps RNN hidden states to character probabilities (softmax over vocabulary + blank).

## Step-by-Step Breakdown

### Step 1: Audio Feature Extraction
Generate synthetic audio (sine waves combining multiple frequencies to simulate speech-like signals). Implement the full Mel spectrogram pipeline: framing, windowing, FFT, Mel filterbank construction, and log compression. This is the most numerically intensive step — get it right and the rest follows.

### Step 2: CTC Loss (Forward Algorithm)
Implement the CTC forward algorithm using dynamic programming. This computes the probability of a target sequence given the model's output probabilities, marginalized over all valid alignments. The key is handling the blank token and the "no consecutive duplicate" constraint correctly.

### Step 3: Neural Network Model
Build a simple Conv → BiGRU → Linear model using only NumPy. The forward pass processes Mel spectrogram frames through convolution (for local feature extraction), bidirectional GRU (for temporal context), and a linear layer (for character probabilities).

### Step 4: Greedy Decoder
Implement the CTC greedy decoder: take argmax at each time step, collapse consecutive duplicates, remove blanks, and map indices back to characters.

### Step 5: End-to-End Pipeline
Wire everything together: audio → Mel spectrogram → model → CTC decode → text. Demonstrate with synthetic examples that show the pipeline working.

## Learning Objectives

- Implement Mel spectrogram extraction from first principles (FFT, filterbanks, log compression)
- Understand the CTC alignment problem and implement the forward algorithm
- Build a sequence-to-sequence ASR model architecture (Conv + RNN + Linear)
- Implement greedy CTC decoding with blank removal and deduplication
- Connect audio signal processing to deep learning for speech recognition

## Going Deeper

- **Beam search decoding**: Instead of greedy argmax, maintain top-K hypotheses at each step. Dramatically improves accuracy.
- **Language model integration**: Shallow fusion — combine CTC scores with an external language model during beam search. This is how production systems handle homophones and ambiguous acoustics.
- **Attention-based models**: Replace CTC with encoder-decoder attention (like Whisper). More flexible but requires more data and compute.
- **Streaming ASR**: Modify the architecture for real-time processing — use unidirectional RNNs and chunked attention instead of bidirectional models.
- **Data augmentation**: SpecAugment (masking time/frequency bands in spectrograms) is simple and extremely effective for ASR training.
- **Tokenization**: Character-level models are simple but slow at inference. Production systems use BPE/SentencePiece for subword tokenization.
