import os
def write_to_log(dir: str, filename: str,content: str,mode='w'):
  if not os.path.exists( dir ):
    os.makedirs(dir)
  if os.path.exists( os.path.join(dir,filename) ):
    mode = 'a'
  with open(os.path.join(dir,filename),mode) as f:
    f.write(content)
