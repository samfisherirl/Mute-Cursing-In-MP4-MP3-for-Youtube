from cryptography.fernet import Fernet
import csv

# Constants for filenames
ORIGINAL_CSV_FILENAME = 'curse_words.csv'
ENCRYPTED_CSV_FILENAME = ORIGINAL_CSV_FILENAME + '.enc'
DECRYPTED_CSV_FILENAME = 'dec_' + ORIGINAL_CSV_FILENAME
KEY_FILENAME = 'filekey.key'

# Function to generate a key and save it
def write_key():
    key = Fernet.generate_key()
    with open(KEY_FILENAME, 'wb') as key_file:
        key_file.write(key)
    return key

# Function to load the key
def load_key():
    return open(KEY_FILENAME, 'rb').read()

# Encrypt the CSV file
def encrypt_csv(filename, key):
    f = Fernet(key)
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        encrypted_rows = [f.encrypt(str(row).encode()) for row in reader]
    
    with open(ENCRYPTED_CSV_FILENAME, 'wb') as file:
        for row in encrypted_rows:
            file.write(row + b'\n')

# Decrypt the encrypted CSV file
def decrypt_csv(filename, key):
    f = Fernet(key)
    with open(filename, 'rb') as file:
        encrypted_rows = file.readlines()
    
    decrypted_rows = [f.decrypt(row).decode() for row in encrypted_rows]
    curse_words = []
    with open(DECRYPTED_CSV_FILENAME, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in decrypted_rows:
            # Convert string representation of list back to a list
            row_list = eval(row)
            if len(row_list) > 0:
                curse_words.append(row_list[0])
                writer.writerow(row_list)
    return curse_words

# Main process
if __name__ == '__main__':
    # Generate and write a new key
    # key = write_key()
    
    # Encrypt the CSV file
    # encrypt_csv(ORIGINAL_CSV_FILENAME, load_key())
    
    # Decrypt the file
    curse_words = decrypt_csv(ENCRYPTED_CSV_FILENAME, load_key())
    
    # Read the decrypted content
    with open(DECRYPTED_CSV_FILENAME, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)