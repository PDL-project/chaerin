import csv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
ai2thor_dir = os.path.dirname(script_dir)
file_path = os.path.join(ai2thor_dir, "objects_actions.tsv")


class EnvObjects:
    def __init__(self):

        self.attribute_objects = {
            'Openable': [],
            'Pickupable': [],
            'On/Off': [],
            'Receptacle': [],
            'Fillable': [],
            'Sliceable': [],
            'Cookable': [],
            'Breakable': [],
            'Dirty': [],
            'UsedUp': []
        }

        with open(file_path, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')

            for row in reader:
                for attribute in self.attribute_objects:
                    if row[attribute] == 'yes':
                        self.attribute_objects[attribute].append(row['Object Type'])



    def get_object_with_action(self, action):
        objects = self.attribute_objects.get(action, [])
        return objects
    

if __name__ == "__main__":
    getter = EnvObjects()
    response = getter.get_object_with_action('Pickupable')
    print(response)


            