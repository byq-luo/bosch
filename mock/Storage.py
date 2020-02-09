import os
class Storage:
  def recursivelyFindVideosInFolder(self, rootFolder: str):
    rootFolder = 'mock/fakeVideos'

    videoPaths = []
    numFilesFound = 0
    for folderPath,subdirs,files in self.__recursivelyWalk(rootFolder):
      for file in files:
        if numFilesFound >= 30:
          return videoPaths
        name, ext = os.path.splitext(file)
        # fake video files
        if ext == '.txt':
          file = file.replace('_labels', '_m0')
          file = file.replace('.txt', '.avi')
          videoPaths += [folderPath + '/' + file]
          numFilesFound += 1
    return videoPaths

  def __recursivelyWalk(self, rootFolder):
    return os.walk(rootFolder)
