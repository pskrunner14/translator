import os
import time
import random
import argparse
import logging
import configparser

import torch

from torch import nn
from torch import optim
from tqdm import tqdm

from utils import init_cuda, prepare_data, tensors_from_pair, time_since, save_pickle
from model import EncoderRNN, AttnDecoderRNN
from evaluate import evaluate_randomly

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 15


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration')

    parser.add_argument('--epochs', type=int, default=30, dest='epochs',
                        help='Number of iterations for training')

    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size',
                        help='Batch size for one epoch in training')

    parser.add_argument('--lr', type=float, default=0.001, dest='lr',
                        help='Initial learning rate')

    parser.add_argument('--num-layers', type=int, default=2, dest='num_layers',
                        help='Number of layers in the RNN models')

    parser.add_argument('--hidden-size', type=int, default=512, dest='hidden_size',
                        help='Number of hidden units in each layer of RNN')

    parser.add_argument('--dropout-rate', type=float, default=0.5, dest='dropout_p',
                        help='Dropout rate for the decoder RNN')

    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5, dest='teacher_forcing_ratio',
                        help='Probability of teacher forcing the training of the model')

    parser.add_argument('--model-name', type=str, default='eng_deu.2_512_adamax', dest='model_name',
                        help='Name for the model')

    parser.add_argument('--save-every', type=int, default=5, dest='save_every',
                        help='Iteration interval after which to save the model')

    parser.add_argument('--log-level', type=str, default='info', dest='log_level',
                        help='Logging level')

    return parser.parse_args()


def optimize(training_tensors, encoder, decoder, encoder_optimizer,
             decoder_optimizer, loss_function, teacher_forcing_ratio=0.5, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    target_lens = 0

    for input_tensor, target_tensor in training_tensors:

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        target_lens += target_length

        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(max_length,
                                      encoder.hidden_size, device=torch.device('cuda'))

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = torch.tensor(
            [[SOS_TOKEN]], device=torch.device('cuda'))
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input,
                                                            decoder_hidden, encoder_outputs)
                loss += loss_function(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input,
                                                            decoder_hidden, encoder_outputs)
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += loss_function(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_TOKEN:
                    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_lens


def train_epochs(encoder, decoder, pairs, input_lang, output_lang, epochs=20,
                 batch_size=32, save_every=5, lr=0.001, teacher_forcing_ratio=0.5,
                 lr_step_divisor=2, lr_step_gamma=0.1, model_name='autoencoder'):

    start = time.time()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    encoder_lr_scheduler = optim.lr_scheduler.StepLR(
        encoder_optimizer, step_size=epochs // lr_step_divisor, gamma=lr_step_gamma)
    decoder_lr_scheduler = optim.lr_scheduler.StepLR(
        decoder_optimizer, step_size=epochs // lr_step_divisor, gamma=lr_step_gamma)

    loss_function = nn.NLLLoss()

    for epoch in range(1, epochs + 1):

        total_loss = 0
        random.shuffle(pairs)

        if len(pairs) % batch_size == 0:
            iters = len(pairs) // batch_size
        else:
            iters = (len(pairs) // batch_size) + 1

        for i in tqdm(range(0, len(pairs), batch_size), total=iters,
                      desc='Epoch {}/{}'.format(epoch, epochs), leave=False):

            if i + batch_size <= len(pairs):
                training_tensors = [tensors_from_pair(pair, input_lang,
                                                      output_lang) for pair in pairs[i: i + batch_size]]
            else:
                training_tensors = [tensors_from_pair(pair, input_lang,
                                                      output_lang) for pair in pairs[i:]]

            loss = optimize(training_tensors, encoder, decoder,
                            encoder_optimizer, decoder_optimizer, loss_function,
                            teacher_forcing_ratio=teacher_forcing_ratio)
            total_loss += loss

            encoder_lr_scheduler.step()
            decoder_lr_scheduler.step()

        total_loss_avg = total_loss / iters
        total_loss = 0
        print('Epoch {}/{}: {:.2f}% | Loss: {:.2f} | {}'
              .format(epoch, epochs, (epoch / epochs) * 100,
                      total_loss_avg, time_since(start, epoch / epochs)))

        if epoch % save_every == 0:
            logging.info('Saving models on epoch {}'.format(epoch))
            torch.save(encoder.state_dict(), 'models/{}/{}_{}.encoder'
                       .format(model_name, epoch, model_name))
            torch.save(decoder.state_dict(), 'models/{}/{}_{}.decoder'
                       .format(model_name, epoch, model_name))


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
    config['rnn'] = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout_p': args.dropout_p
    }

    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, args.log_level.upper()))

    if not os.path.isdir('models/{}'.format(args.model_name)):
        os.mkdir('models/{}'.format(args.model_name))

        input_lang, output_lang, train_pairs, test_pairs = \
            prepare_data('eng', 'deu', True)
        print(random.choice(train_pairs))

        encoder, decoder = create_models(config['rnn'], input_lang.n_words,
                                         output_lang.n_words)

        logging.info('Saving model configuration for evaluation...')
        with open('models/{}/{}.cfg'.format(args.model_name, args.model_name), 'w') as config_file:
            config.write(config_file)

        logging.info('Saving data...')
        save_pickle((input_lang, output_lang, train_pairs, test_pairs),
                    'models/{}/{}.data'.format(args.model_name, args.model_name))
    else:
        try:
            files = os.listdir('models/{}'.format(args.model_name))

            logging.info('Reading configuration...')
            config_file = 'models/{}/{}'.format(args.model_name,
                                                list(fileter(lambda x: '.cfg' in x, files))[0])
            config.read(config_file)

            logging.info('Loading data...')
            data_file = 'models/{}/{}'.format(args.model_name,
                                              list(filter(lambda x: '.data.pkl' in x, files))[0][:-4])
            input_lang, output_lang, train_pairs, test_pairs = load_pickle(
                data_file)

            logging.info('Loading latest pretrained checkpoint models...')
            iters = [int(x.split('_')[0])
                     for x in files if '.encoder' in x or '.decoder' in x]
            max_iter = str(max(iters))

            encoder_path = 'models/{}/{}'.format(args.model_name,
                                                 list(filter(lambda x: x.startswith(max_iter) and '.encoder' in x, files))[0])
            decoder_path = 'models/{}/{}'.format(args.model_name,
                                                 list(fileter(lambda x: x.startswith(max_iter) and '.decoder' in x, files))[0])

            encoder, decoder = create_models(config['rnn'], input_lang.n_words,
                                             output_lang.n_words)
            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))

        except Exception as e:
            logging.debug(str(e))
            logging.critical(
                'Files required for retraining not found! Please delete the model dir.')
            exit(0)

    logging.info('Training models...')
    train_epochs(encoder, decoder, train_pairs, input_lang, output_lang,
                 epochs=args.epochs, batch_size=args.batch_size, save_every=args.save_every,
                 lr=args.lr, teacher_forcing_ratio=args.teacher_forcing_ratio,
                 model_name=args.model_name)

    logging.info('Saving trained models...')
    torch.save(encoder.state_dict(), 'models/{}/{}.encoder'
               .format(args.model_name, args.model_name))
    torch.save(decoder.state_dict(), 'models/{}/{}.decoder'
               .format(args.model_name, args.model_name))

    logging.info('Evaluating models...')
    evaluate_randomly(test_pairs, encoder, decoder, input_lang, output_lang)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')
