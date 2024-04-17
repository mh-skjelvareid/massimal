import pytest
import random
import numpy as np
from massimal.annotation import class_indices_from_hierarchy

@pytest.fixture
def example_class_indices_and_hierarchy():
    class_indices = {'Vegetation':1,'Grass':2,'Trees':3,'Oak':4,'Birch':5,
            'Rock':6,'Bedrock':7,'Cobble':8,'Buildings':9,'Test':10}
    class_hierarchy = {'Vegetation':{'Grass':[],'Trees':['Oak','Birch']},
        'Rock':['Bedrock','Cobble'],'Buildings':[]}
    return (class_indices,class_hierarchy)

@pytest.fixture
def example_mask_9_classes():
    mask = np.zeros(shape=(100,100),dtype=np.uint8)
    for i in range(9):
        rc = (i//3)*30 + 20
        cc = (i%3)*30 + 20
        rs = round(random.random()*10) + 5
        cs = round(random.random()*10) + 5
        mask[(rc-rs):(rc+rs),(cc-cs):(cc+cs)] = i+1
    return mask

def test_class_indices_from_hierarchy(example_class_indices_and_hierarchy):
    class_indices,class_hierarchy = example_class_indices_and_hierarchy
    grouped_class_indices = class_indices_from_hierarchy(
        class_hierarchy,
        class_indices,
        ['Grass','Trees','Rock'])
    assert grouped_class_indices == {'Grass': {2}, 'Trees': {3, 4, 5}, 'Rock': {6, 7, 8}}