class Vehicle:
  def __init__(self, box, _id):
    self.id = _id
    self.box = box
  def __repr__(self):
    return str(self.id) + '  ' + str(self.box)