def log_to_file(file_name, line):
    filepath = 'results/'+file_name
    file = open(filepath, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()
