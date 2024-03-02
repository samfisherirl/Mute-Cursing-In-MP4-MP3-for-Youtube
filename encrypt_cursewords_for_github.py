from cryptography.fernet import Fernet
import csv

# Function to generate a key and save it
def write_key():
    key = Fernet.generate_key()
    with open('filekey.key', 'wb') as key_file:
        key_file.write(key)
    return key

# Function to load the key
def load_key():
    return open('filekey.key', 'rb').read()

# Function to encrypt the CSV file
def encrypt_file(filename, key):
    f = Fernet(key)
    with open(filename, 'rb') as file:
        file_data = file.read()
    encrypted_data = f.encrypt(file_data)
    with open(filename + '.enc', 'wb') as file:
        file.write(encrypted_data)

# Function to decrypt the CSV file
def decrypt_file(filename, key):
    f = Fernet(key)
    with open(filename, 'rb') as file:
        encrypted_data = file.read()
    decrypted_data = f.decrypt(encrypted_data)
    with open('dec_' + filename, 'wb') as file:
        file.write(decrypted_data)

# Main process
if __name__ == '__main__':
    # Generate and write a new key
    key = write_key()

    # Encrypt the CSV file
    encrypt_file('curse_words.csv', key)

    # Decrypt the file
    decrypt_file('curse_words.csv.enc', key)

    # Read the decrypted content
    with open('dec_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
