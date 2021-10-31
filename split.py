import os
import shutil

config = '''
nav:
'''

for root, dirs, files in os.walk(".", topdown=False):
  if root != '.': continue
  # print(files)
  
  try:
    shutil.rmtree('tmp')
  except FileNotFoundError:
    print('no tmp folder')
  os.mkdir('tmp')

  for file in files:
    if '.md' not in file: continue
    if 'README' in file: continue
    # print(file)
    shutil.copy(file, 'tmp/' + file)
      
    with open('./tmp/' + file, 'r', encoding='utf-8') as f:
      lst = f.read().split('\n' + r'### ')
      folder = lst[0].split('\n')[0]
      while folder[0] in ['#', ' ']: 
        folder = folder[1:]
      # folder name done

      os.mkdir('tmp/' + folder)
      config += '  - ' + folder + ':\n'

      for str in lst[1:]:
        filename = str.split('\n')[0].strip()
        with open('tmp/' + folder + '/' + filename + '.md', 'w', encoding='utf-8') as w:
          w.write(str)
        # print(filename)
        # print('---'*10)
        config += '    - ' + 'tmp/' + folder + '/' + filename + '.md\n'
    os.remove('tmp/' + file)
  # config done

  print('All clear')
  print('将tmp/ 放入OI-wiki/docs\n将tmp/mkdocs.yml内容覆盖OI-wiki/mkdocs.yml')
  with open('tmp/mkdocs.yml', 'w', encoding='utf-8') as w:
    w.write(config)
  

