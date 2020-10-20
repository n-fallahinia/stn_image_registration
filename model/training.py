"""Tensorflow utility functions for training"""

import os
import datetime
import time

from tqdm import tqdm
import tensorflow as tf
import numpy as np
from time import sleep

from utils.utils import save_dict_to_json

class Train_and_Evaluate():
    
    def __init__(self, train_model_spec, train_ds, eval_ds, log_dir):

        self.train_model_spec = train_model_spec
        self.train_ds =  train_ds
        self.eval_ds = eval_ds
        self.log_dir = log_dir

        # Get relevant graph operations or nodes needed for training
        self.model = self.train_model_spec['model']
        self.loss_object = self.train_model_spec['loss']
        self.opt = self.train_model_spec['opt']

        self.metrics = train_model_spec['metrics']
        self.train_loss = self.metrics['train_loss']
        self.train_accuracy_mse= self.metrics['train_MSE']
        self.train_accuracy_kld= self.metrics['train_KLD']

        self.test_accuracy_mse = self.metrics['test_MSE']
        self.test_accuracy_kld = self.metrics['test_KLD']

    @tf.function
    def train_step(self, x_train, y_train):
        """Train the model on batches
        Args:
            model_spec: (dict) contains the graph operations or nodes needed for training
            writer: (tf.summary.FileWriter) writer for summaries
            x_train: training images
            y_train: training measured forces
        """
        # keep track of our gradients
        with tf.GradientTape() as tape:
            # make a prediction using the model and then calculate the loss
            logits = self.model(x_train, training=True) #!!!!! FIX THE NEGATIVE VALUE ISSUE !!!!
            loss = self.loss_object(y_train, logits)
        # calculate the gradients using our tape and then update the model weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # write metices to writer for summary use
        self.train_accuracy_mse.update_state(y_train, logits)
        self.train_accuracy_kld.update_state(y_train, logits)
        self.train_loss.update_state(loss)

        return loss, logits

    @tf.function
    def test_step(self, x_test, y_test):
        """Testing the model on batches
        Args:
            model_spec: (dict) contains the graph operations or nodes needed for training
            x_train: training images
            y_train: training measured forces
        """
        y_test_pred = self.model(x_test, training=False)
        loss = self.loss_object(y_test, y_test_pred)

        # write metices to writer for summary use
        self.test_accuracy_mse.update_state(y_test, y_test_pred)
        self.test_accuracy_kld.update_state(y_test, y_test_pred)

        return loss

    def train_and_eval(self, params, restore_from=None):
        """Train the model and evaluate every epoch.
        Args:
            train_model_spec: (dict) contains the graph operations or nodes needed for training
            params: (Params) contains hyperparameters of the model.
                    Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
            train_ds: training dataset
            eval_ds: evaluation dataset
            log_dir: directory for log
            restore_from: (string) directory or file containing weights to restore the graph
        """
        # set up the train summary writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(self.log_dir, current_time , 'train_summaries')
        eval_log_dir = os.path.join(self.log_dir, current_time , 'eval_summaries')

        checkpoint_dir = os.path.join(self.log_dir, current_time, "training_checkpoints", 'ckpt') 
        model_dir = os.path.join(self.log_dir, current_time)

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

        begin_at_epoch = 0
        best_eval_acc = 100.0

    # TRAINING MAIN LOOP
    # ----------------------------------------------------------------------
        print("[INFO] training started ...")
        # loop over the number of epochs
        epochStart = time.time()
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):

            step = 0
            # sys.stdout.flush()
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps_train = int(np.ceil(params.train_size / params.batch_size))
            num_steps_eval = int(np.ceil(params.eval_size / params.batch_size))  
            # Use tqdm for progress bar
            with tqdm(total=num_steps_train, desc="[INFO] Epoch {0:d}".format(epoch + 1)) as pbar:
                # loop over the data in batch size increments
        # ----------------------------------------------------------------------
        # TRAIN SESSION
                for x_train, y_train in self.train_ds.take(num_steps_train):
                    train_loss, logits = self.train_step(x_train, y_train)
                    # Log the loss in the tqdm progress bar
                    sleep(0.1)
                    # Display metrics at the end of each epoch.
                    metrics = {
                        "Train_MSE": '{:04.2f}'.format(self.train_accuracy_mse.result().numpy()),
                        "Train_Loss": '{:04.2f}'.format(self.train_loss.result().numpy())
                    }
                    pbar.set_postfix(metrics)
                    pbar.update()

                    # record train summary for tensor board
                    # if 0 < step < 30:
                    with train_summary_writer.as_default():
                        tf.summary.image('training images', x_train, step=epoch + step + 1, max_outputs=5)
                        tf.summary.image('logit images', logits, step=epoch + step + 1, max_outputs=5)
                        tf.summary.image('label images', y_train, step=epoch + step + 1, max_outputs=5)
                    step = step +1

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=epoch + 1)
                    tf.summary.scalar('mse', self.train_accuracy_mse.result(), step=epoch + 1)
        # ----------------------------------------------------------------------
        # EVALUATION SESSION
                # loop over the eval data in batch size increments
                for x_eval, y_eval in self.eval_ds.take(num_steps_eval):
                    eval_loss = self.test_step(x_eval, y_eval)
                    # Display metrics at the end of each epoch.
                    metrics["Eval_MSE"] = '{:04.2f}'.format(self.test_accuracy_mse.result().numpy())
                    pbar.set_postfix(metrics)
                pbar.close()
                # record train summary for tensor board
                with eval_summary_writer.as_default():
                    tf.summary.scalar('mse', self.test_accuracy_mse.result(), step=epoch + 1)
        # ----------------------------------------------------------------------
            metrics["Epoch"] = '{0:d}'.format(epoch + 1)
            # If best_eval, save the model at best_save_path 
            eval_acc = self.test_accuracy_mse.result().numpy()
            if params.save_model:
                if eval_acc <= best_eval_acc:
                    # Store new best accuracy
                    best_eval_acc = eval_acc
                    # Save weights
                    best_save_path = os.path.join(model_dir, "model_{0:d}".format(epoch + 1))
                    tf.keras.models.save_model(self.model, best_save_path, save_format = "h5")
                    print("[INFO] Found new best accuracy, saving in {}".format(best_save_path))
                    # Save best eval metrics in a json file in the model directory
                    best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
                    save_dict_to_json(metrics, best_json_path)

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
        # ----------------------------------------------------------------------
            # Reset training metrics at the end of each epoch
            self.train_loss.reset_states()
            self.train_accuracy_mse.reset_states()
            self.train_accuracy_kld.reset_states()

            self.test_accuracy_mse.reset_states()
            self.test_accuracy_kld.reset_states()
        # end of train and eval
        # show timing information for the epoch
        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0
        print("[INFO] Took {:.4} minutes".format(elapsed))
    # ----------------------------------------------------------------------
        if params.save_model:
            reconstructed_best_model = tf.keras.models.load_model(best_save_path)
            reconstructed_best_model.compile(optimizer= self.opt, loss= self.loss_object)
            best_final_path = os.path.join(model_dir, "best_full_model_path")
            tf.saved_model.save(reconstructed_best_model, best_final_path)
            print("[INFO] Final model save in {}".format(best_final_path))
        
        print("[INFO] Training done and log saved in {} ".format(model_dir))

