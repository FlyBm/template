import os
import shutil

config = '''
nav:
'''

for root, dirs, files in os.walk(".", topdown=False):
  if root != '.': continue

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

    newFileName = 'new_' + file  
    
    with open('./tmp/' + newFileName, 'w+', encoding='utf-8') as new_f:
      with open('./tmp/' + file, 'r', encoding='utf-8') as f:

        lines = f.read().split('\n')

        flag = True
        tmp = ''
        for line in lines:
          if flag:
            new_f.write(line + '\n')
          else:
            tmp += line + '\n'
          # print(line)
          if '```' in line:
            if len(tmp) > 1:
              with open('./tmp/a.cpp', 'w+', encoding='utf-8') as c:
                c.write(tmp + '\n')
              # os.system('pwd')
              os.system('clang-format -style=google ./tmp/a.cpp > ./tmp/b.cpp')
              with open('./tmp/b.cpp', 'r', encoding='utf-8') as c:
                new_f.write(c.read())
              flag = True
            else:
              flag = False
            tmp = ''
          
    shutil.move('tmp/' + newFileName, 'tmp/' + file)
    # shutil.rmtree('tmp/' + newFileName)

    # break



  shutil.rmtree('tmp/a.cpp')
  shutil.rmtree('tmp/b.cpp')

  # os.remove('tmp/' + file)
  # config done

  print('All clear')
  

