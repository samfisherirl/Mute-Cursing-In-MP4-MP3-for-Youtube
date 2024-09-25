from cryptography.fernet import Fernet
import csv


def generate_key():
    """Generates a key and saves it into a file"""
    key = Fernet.generate_key()
    with open("key.txt", "wb") as key_file:
        key_file.write(key)
    return key


def load_key():
    """Loads the key from the current directory named `key.txt`"""
    return open("key.txt", "rb").read()


def encrypt_csv_file(csv_filename):
    """Encrypts a CSV file and saves the encryption key and the encrypted content in separate files"""
    key = generate_key()  # Generates and saves the key
    f = Fernet(key)

    # Read the contents of the CSV file
    with open(csv_filename, "rb") as file:
        file_data = file.read()

    # Encrypt the data
    encrypted_data = f.encrypt(file_data)

    # Write the encrypted data to a new file
    with open(csv_filename + ".encrypted", "wb") as file:
        file.write(encrypted_data)
    print("Encryption done!")


def decrypt_csv_file(encrypted_filename, output_filename):
    """Decrypts an encrypted file with the key from `key.txt` and writes it back to a CSV file"""
    key = load_key()  # Load the previously generated key
    f = Fernet(key)

    # Read the encrypted data
    with open(encrypted_filename, "rb") as file:
        encrypted_data = file.read()

    # Decrypt data
    decrypted_data = f.decrypt(encrypted_data)

    # Write the decrypted data back to a CSV file
    with open(output_filename, "wb") as file:
        file.write(decrypted_data)
    print("Decryption done!")


# Example usage