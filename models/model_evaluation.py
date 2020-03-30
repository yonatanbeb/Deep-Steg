import sys
sys.path.append('../')
from encode_dataset import encoded_x_train_clr0, encoded_x_train_clr1, x_train
from encode_dataset import encoded_x_test_clr0, encoded_x_test_clr1, x_test
from encode_dataset import encoded_y_train_clr0, encoded_y_train_clr1, y_train, y_test
from models.auto_encoder import auto_encoder
from models.classifier import classifier


########################################################################################################################

print('Auto Encoder: ')
auto_encoded_clr0, _ = auto_encoder(encoded_x_train_clr0, x_train, encoded_x_test_clr0, x_test, '0', False, True)
auto_encoded_clr1, _ = auto_encoder(encoded_x_train_clr1, x_train, encoded_x_test_clr1, x_test, '1', False, True)

print('\n')

print('Classifier: ')
print('\tNo Clearance: ')
classifier(x_train, y_train, x_test, y_test, False, True)
print('\tClearance Level 0 Evaluation: ')
classifier(auto_encoded_clr0, encoded_y_train_clr0, auto_encoded_clr0, encoded_y_train_clr0, False, True)
print('\tClearance Level 1 Evaluation: ')
classifier(auto_encoded_clr1, encoded_y_train_clr1, auto_encoded_clr1, encoded_y_train_clr1, False, True)

########################################################################################################################
