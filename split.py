import os
import shutil

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
    print(file)
    shutil.copy(file, 'tmp/' + file)
    # with open('./tmp/' + file, 'r', encoding='utf-8') as f:
      

    with open('./tmp/' + file, 'r', encoding='utf-8') as f:
      # print(f.read().split('### '))
      lst = f.read().split('\n' + r'### ')
      # print(len(lst))
      folder = lst[0].split('\n')[0]
      while folder[0] in ['#', ' ']: 
        folder = folder[1:]
      # print(folder)
      # folder name done
      
      os.mkdir('tmp/' + folder)
      for str in lst[1:]:
        filename = str.split('\n')[0]
        # os.
        with open('tmp/' + folder + '/' + filename + '.md', 'w', encoding='utf-8') as w:
          w.write(str)
        # print(filename)
        # print('---'*10)
    # break
