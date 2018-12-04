import os
import random
import argparse
import configparser
import logging

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import init_cuda, tensor_from_sentence, load_pickle
from model import EncoderRNN, AttnDecoderRNN

plt.switch_backend('agg')

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 15


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluation Configuration')

    parser.add_argument('--num-tests', type=int, default=10, dest='num_tests',
                        help='Number of evaluation tests')

    parser.add_argument('--model-name', type=str, default='lstm3_bi_sgd', dest='model_name',
                        help='Name for the model')

    parser.add_argument('--log-level', type=str, default='info', dest='log_level',
                        help='Logging level')

    return parser.parse_args()


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):

    with torch.no_grad():

        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(max_length,
                                      encoder.hidden_size, device=torch.device('cuda'))

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor(
            [[SOS_TOKEN]], device=torch.device('cuda'))
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
        output_words, _ = evaluate(
            encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('Predicted: {}\n'.format(output_sentence))


def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence, encoder, decoder, input_lang, output_lang):
    output_words, attentions = evaluate(encoder, decoder,
                                        input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


def create_models(config, in_words, out_words):
    logging.info('Creating models...')
    encoder = EncoderRNN(in_words, int(config['hidden_size']),
                         num_layers=int(config['num_layers'])).cuda()

    decoder = AttnDecoderRNN(int(config['hidden_size']), out_words,
                             num_layers=int(config['num_layers']),
                             dropout_p=float(config['dropout_p'])).cuda()
    return encoder, decoder


def main():
    init_cuda()

    args = parse_arguments()
    config = configparser.ConfigParser()

    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, args.log_level.upper()))

    try:
        files = os.listdir('models/{}'.format(args.model_name))

        logging.info('Reading configuration...')
        config_file = 'models/{}/{}'.format(args.model_name,
                                            list(fileter(lambda x: '.cfg' in x, files))[0])
        config.read(config_file)

        logging.info('Loading data...')
        data_file = 'models/{}/{}'.format(args.model_name,
                                          list(filter(lambda x: '.data.pkl' in x, files))[0][:-4])
        input_lang, output_lang, _, test_pairs = load_pickle(data_file)

        logging.info('Loading pretrained checkpoint models...')

        encoder_path = 'models/{}/{}'.format(args.model_name,
                                             list(filter(lambda x: '.encoder' in x, files))[0])
        decoder_path = 'models/{}/{}'.format(args.model_name,
                                             list(fileter(lambda x: '.decoder' in x, files))[0])

        encoder, decoder = create_models(config['rnn'], input_lang.n_words,
                                         output_lang.n_words)
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))

    except Exception as e:
        logging.debug(str(e))
        logging.critical(
            'Files critical for evaluating not found! Please retrain the model.')
        exit(0)

    logging.info('Evaluating models...')
    evaluate_randomly(test_pairs, encoder, decoder,
                      input_lang, output_lang, n=args.num_tests)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')
