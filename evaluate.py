import sys
import random

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from configparser import ConfigParser

from utils import get_torch_device, prepare_data, tensor_from_sentence, load_pickle
from network import EncoderRNN, AttnDecoderRNN

plt.switch_backend('agg')

config = ConfigParser()
config.read('config.cfg')

EOS_TOKEN = int(config['model']['eos_token'])
SOS_TOKEN = int(config['model']['sos_token'])
MAX_LENGTH = int(config['model']['max_length'])


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        
        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=get_torch_device())
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], 
                encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
            
        decoder_input = torch.tensor([[SOS_TOKEN]], device=get_torch_device())
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            _, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.idx_to_word[topi.item()])
            decoder_input = topi.squeeze().detach()
            
        return decoded_words, decoder_attentions[: di + 1]
    
def evaluate_randomly(pairs, encoder, decoder, input_lang, output_lang, n=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print('Input: {}'.format(pair[0]))
        print('Target: {}'.format(pair[1]))
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('Predicted: {}\n'.format(output_sentence))

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence, encoder, decoder, input_lang, output_lang):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
        
if __name__ == '__main__':

    '''Using CUDA Device'''
    device = get_torch_device()
    device_idx = torch.cuda.current_device()
    device_cap = torch.cuda.get_device_capability(device_idx)
    print('PyTorch: Using {} Device {}:{} with Compute Capability {}.{}'
        .format(str(device).upper(), torch.cuda.get_device_name(device_idx), device_idx, device_cap[0], device_cap[1]))
    
    if len(sys.argv) == 5:
        num_tests, encoder_path, decoder_path, config_path = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    else:
        print('Usage: python evaluate.py [num tests] [encoder] [decoder]')
        exit(0)

    config = ConfigParser()
    config.read('models/{}'.format(config_path))

    input_lang, output_lang, pairs = load_pickle('models/eng-fra.data')
    
    encoder = EncoderRNN(input_lang.n_words, int(config['rnn']['hidden_size']), 
            layer_type=config['rnn']['layer_type'], num_layers=int(config['rnn']['num_layers'])).to(device)
            
    decoder = AttnDecoderRNN(int(config['rnn']['hidden_size']), output_lang.n_words, layer_type=config['rnn']['layer_type'], 
            num_layers=int(config['rnn']['num_layers']), dropout_p=float(config['rnn']['decoder_dropout'])).to(device)

    encoder.load_state_dict(torch.load('models/{}'.format(encoder_path)))
    decoder.load_state_dict(torch.load('models/{}'.format(decoder_path)))

    evaluate_randomly(pairs, encoder, decoder, input_lang, output_lang, n=num_tests)