import os, pickle

# Useful for dependency injection in testing.
class Storage:
  def recursivelyFindVideosInFolder(self, rootFolder: str):
    videoPaths = []
    for folderPath,subdirs,files in self.__recursivelyWalk(rootFolder):
      for file in files:
        name, ext = os.path.splitext(file)
        if ext == '.avi':
          videoPaths += [folderPath + '/' + file]
    return videoPaths

  def __recursivelyWalk(self, rootFolder):
    return os.walk(rootFolder)

  def writeListToFile(self, lines : list, filename: str):
    with open(filename, 'w') as file:
      file.writelines(lines)

  def writeObjsToPkl(self, objs: list, filename: str):
    with open(filename, 'wb') as file:
      pickle.dump(objs, file)
  
  def loadObjsFromPkl(self, filename: str):
    with open(filename, 'rb') as file:
      return pickle.load(file)