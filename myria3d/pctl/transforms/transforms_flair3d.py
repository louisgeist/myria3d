import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_scatter import scatter_sum

lidarhd_class_dictionary = {
    1: ("#d3d3d3", "Non classé"),
    2: ("#a0522d", "Sol"),
    3: ("#b3b94d", "Végétation basse"),
    4: ("#4e9a4e", "Végétation moyenne"),
    5: ("#1f4e1f", "Végétation haute"),
    6: ("#ff0000", "Bâtiment"),
    9: ("#1e90ff", "Eau"),
    17: ("#ffff00", "Pont"),
    64: ("#ff8c00", "Sursol pérenne"),
    65: ("#8b00ff", "Artefact"),
    66: ("#000000", "Points virtuels (modélisation)")
}

LIDARHD_NUM_CLASSES = 10

# As detailed in the docs/datasets.md, labels are expacted to be in the 
# range [0, C], where C is the number of classes. The `C` label is reserved 
# for void/ignored/unlabeled points.
TRAINID = 0
LIDARHD_ID2TRAINID = np.ones(max(lidarhd_class_dictionary.keys())+1)*LIDARHD_NUM_CLASSES
for k,v in lidarhd_class_dictionary.items():
    if v[1] == 'Non classé':
        # LIDARHD_ID2TRAINID[k] = LIDARHD_NUM_CLASSES (void class)
        pass
    else:
        LIDARHD_ID2TRAINID[k] = TRAINID
        TRAINID += 1
        

class Flair3DToConsecutiveLabels(BaseTransform):
    LIDARHD_ID2TRAINID = torch.from_numpy(LIDARHD_ID2TRAINID).long()
    LIDARHD_ID2TRAINID_on_device = None
    
    _NO_REPR = ['LIDARHD_ID2TRAINID_on_device']
    
    def __call__(self, data):
        """
        Remap the Flair3D labels to the range [0, C], where C is the number of 
        classes.
        
        The idx C is reserved for void/ignored/unlabeled points.
        """
        
        device = data.x.device
        
        if getattr(data, 'y_cosia', None) is not None:
            pass # Already as we want (consecutive labels, C for void)
        
        if getattr(data, 'y_lidarhd', None) is not None:
            y_lidarhd = data.y_lidarhd.clone()
            
            if torch.any(y_lidarhd > 66):
                # print(f"Warning: {(y_lidarhd > 66).int().sum()} are idx > 66 in lidarhd labels.")
                y_lidarhd[y_lidarhd > 66] = 1 # to 'non classé' label
                
            # We need LIDARHD_ID2TRAINID on the same device as y_lidarhd for
            # the computation. If we moved LIDARHD_ID2TRAINID to GPU in place,
            # it would be included in __repr__ (used for pre_transform_hash)
            # and cause instability when computing the pretransform hash. So we
            # use a separate LIDARHD_ID2TRAINID_on_device for the actual
            # computation; it is in _NO_REPR and thus excluded from __repr__
            # and from the hash.
            if self.LIDARHD_ID2TRAINID_on_device is None:
                self.LIDARHD_ID2TRAINID_on_device = self.LIDARHD_ID2TRAINID.to(device)

            data.y_lidarhd = self.LIDARHD_ID2TRAINID_on_device[y_lidarhd]
            
        return data
    
class Flair3DRemapLabels(BaseTransform):
    
    def __init__(self, y_definition='coarse_lidarhd'):
        self.FLAIR3D_COARSE_NUM_CLASSES = 3
    
        self.BUILDING_COARSE_1 = 0
        self.SOIL_COARSE_1 = 1
        self.VEGETATION_COARSE_1 = 2
    
        
        ### ---- COSIA ---- ###
        self.COSIA_2_FLAIR3D = torch.tensor([
            self.BUILDING_COARSE_1, # Building
            self.BUILDING_COARSE_1, # Greenhouse
            self.FLAIR3D_COARSE_NUM_CLASSES, # Swimming pool
            self.SOIL_COARSE_1, # Impervious surface
            self.SOIL_COARSE_1, # Pervious surface
            self.SOIL_COARSE_1, # Bare soil
            self.FLAIR3D_COARSE_NUM_CLASSES, # Water
            self.FLAIR3D_COARSE_NUM_CLASSES, # Snow
            self.SOIL_COARSE_1, # Herbaceous vegetation
            self.SOIL_COARSE_1, # Agricultural land
            self.SOIL_COARSE_1, # Plowed land
            self.VEGETATION_COARSE_1, # Vineyard
            self.VEGETATION_COARSE_1, # Deciduous
            self.VEGETATION_COARSE_1, # Coniferous
            self.VEGETATION_COARSE_1, # Brushwood
            self.FLAIR3D_COARSE_NUM_CLASSES, # Clear cut 
            self.FLAIR3D_COARSE_NUM_CLASSES, # Ligneous 
            self.FLAIR3D_COARSE_NUM_CLASSES, # Mixed
            self.FLAIR3D_COARSE_NUM_CLASSES, # Undefined
        ])
        
        ### ---- LIDARHD ---- ###
        self.LIDARHD_2_FLAIR3D = torch.tensor([
            self.SOIL_COARSE_1, # Soil
            self.SOIL_COARSE_1, # Végétation basse
            self.SOIL_COARSE_1, # Végétation moyenne
            self.VEGETATION_COARSE_1, # Végétation haute
            self.BUILDING_COARSE_1, # Bâtiment
            self.FLAIR3D_COARSE_NUM_CLASSES, # Eau
            self.BUILDING_COARSE_1, # Pont
            self.BUILDING_COARSE_1, # Sursol pérenne
            self.FLAIR3D_COARSE_NUM_CLASSES, # Artefact
            self.FLAIR3D_COARSE_NUM_CLASSES, # Points virtuels (modélisation)
            self.FLAIR3D_COARSE_NUM_CLASSES,
        ])
        
        ### ---- Disagreements color mapping ---- ###
        self.DISAGREEMENT_COLORS = torch.tensor(
            [[1,0,0],  # red for disagreement
            [0,1,0]],  # green for agreement
            )
        
        self.DISAGREEMENT_COLORS_CLASS_OF_INTEREST = torch.tensor(
            [[0.4, 0.75, 0.45],  # green for agreement
            [0.65, 0.45, 0.75],  # purple for cosia but not lidarhd
            [0.9, 0.6, 0.35],  # orange for lidarhd but not cosia
            [0.65, 0.65, 0.65]],  # grey for no class of interest
            )
        
        self.mapped_to_device = False
        
        self.y_definition = y_definition
    

    _NO_REPR = ['mapped_to_device',
                'COSIA_2_FLAIR3D_on_device', 
                'LIDARHD_2_FLAIR3D_on_device', 
                
                'DISAGREEMENT_COLORS_on_device', 
                'DISAGREEMENT_COLORS_CLASS_OF_INTEREST_on_device',
                
                'DISAGREEMENT_COLORS',
                'DISAGREEMENT_COLORS_CLASS_OF_INTEREST',
                ]
    
    def __call__(self, data):
        """
        Remap the cosia and lidarhd labels to the coarse labels.
        
        Also compute the disagreements colors and the fixed labels.
        
        New attributes:
        - y_cosia_coarse: the cosia labels remapped to the coarse labels
        - y_lidarhd_coarse: the lidarhd labels remapped to the coarse labels
        - y_colors: the colors of the disagreements
        - y_fixed: the fixed labels
        """
        self.map_to_device(data.x.device)
        
        cosia_available = getattr(data, 'y_cosia', None) is not None
        lidarhd_available = getattr(data, 'y_lidarhd', None) is not None
        
        if cosia_available:
            data.y_cosia_coarse = self.remap(data.y_cosia, self.COSIA_2_FLAIR3D_on_device)
            y_cosia_coarse_majority = self.histogram_to_onehot(data.y_cosia_coarse)
            
        if lidarhd_available:
            data.y_lidarhd_coarse = self.remap(data.y_lidarhd, self.LIDARHD_2_FLAIR3D_on_device)
            y_lidarhd_coarse_majority = self.histogram_to_onehot(data.y_lidarhd_coarse)
            
        assert cosia_available and lidarhd_available, "cosia and lidarhd must be available to compute `y_fixed`."
        
        # Compute agreement mask
        y_agreement_mask = (y_cosia_coarse_majority == y_lidarhd_coarse_majority).int()
        data.y_agreement = self.remap(y_agreement_mask, self.DISAGREEMENT_COLORS_on_device)
                
        
        # Compute y_fixed
        if self.y_definition == 'coarse_lidarhd':
            y_fixed = y_lidarhd_coarse_majority.clone()
        elif self.y_definition == 'coarse_cosia':
            y_fixed = y_cosia_coarse_majority.clone()
        elif self.y_definition == 'coarse_intersection':
        
            
            y_fixed = self.FLAIR3D_COARSE_NUM_CLASSES * \
                torch.ones_like(y_cosia_coarse_majority, dtype=torch.long)
            
            y_agreement_mask = y_agreement_mask.bool()
            # Keep only the coarse labels where both agree.
            y_fixed[y_agreement_mask] = y_cosia_coarse_majority[y_agreement_mask]
            
            
        elif self.y_definition == 'rule1':
            y_fixed = y_cosia_coarse_majority.clone()
            
            # Fixing soil labels not on the ground
            # We keep only the soil labels if both agree on it.
            mask = (y_cosia_coarse_majority == self.SOIL_COARSE_1) \
                & (y_lidarhd_coarse_majority != self.SOIL_COARSE_1)
            is_not_void = y_lidarhd_coarse_majority != self.FLAIR3D_COARSE_NUM_CLASSES
            
            y_fixed[mask & is_not_void] = y_lidarhd_coarse_majority[mask & is_not_void]
            
            
        else:
            raise ValueError(f"Invalid y_definition: {self.y_definition}")
        
        # to histogram
        to_histogram = False
        if to_histogram:
            histogram = torch.zeros((y_fixed.shape[0], 
                                    self.FLAIR3D_COARSE_NUM_CLASSES + 1), 
                                    device=data.x.device,
                                    dtype=torch.long)
            
            histogram[torch.arange(y_fixed.shape[0]), y_fixed] = 1
            y_fixed = histogram
        data.y = y_fixed
        
        # Disagreements focused on a class
        class_of_interest = self.BUILDING_COARSE_1
        if class_of_interest is not None:
            
            agreements = (y_cosia_coarse_majority == class_of_interest) \
                & (y_lidarhd_coarse_majority == class_of_interest)
                
            cosia_but_not_lidarhd = (y_cosia_coarse_majority == class_of_interest) \
                & (y_lidarhd_coarse_majority != class_of_interest)
                
            lidarhd_but_not_cosia = (y_cosia_coarse_majority != class_of_interest) \
                & (y_lidarhd_coarse_majority == class_of_interest)
                
            no_class_of_interest = (y_cosia_coarse_majority != class_of_interest) \
                & (y_lidarhd_coarse_majority != class_of_interest)
                
                
            class_of_interest_idx = torch.zeros((data.y_cosia_coarse.shape[0],), 
                                                device=data.x.device,
                                                dtype=torch.long)
            class_of_interest_idx[agreements] = 0
            class_of_interest_idx[cosia_but_not_lidarhd] = 1
            class_of_interest_idx[lidarhd_but_not_cosia] = 2
            class_of_interest_idx[no_class_of_interest] = 3
            
            data[f"class_{class_of_interest}_idx"] = class_of_interest_idx # to easy display of the text
            data[f"class_{class_of_interest}"] = \
                self.DISAGREEMENT_COLORS_CLASS_OF_INTEREST_on_device[class_of_interest_idx]
            
        return data
    
    def remap(self, y, mapping): 
        if y.dim() == 1: # y is a list of labels (y in [0, C]^N)
            return mapping[y]
        
        elif y.dim() == 2: # y is an histogram of labels (shape (N,C))
            return scatter_sum(y, mapping, dim=1)
        else:
            raise ValueError(f"Expected y.dim()=1 or y.dim()=2, got y.dim()={y.dim()} instead")

    def map_to_device(self, device):
        
        # We need the following tensors on the same device as data for
        # the computation. If we moved them to GPU in place,
        # they would be included in __repr__ (used for pre_transform_hash)
        # and cause instability when computing the pretransform hash. So we
        # use a separate <tensor_name>_on_device for the actual
        # computation; which is in _NO_REPR and thus excluded from __repr__
        # and from the hash.
        if not self.mapped_to_device:
            self.COSIA_2_FLAIR3D_on_device = self.COSIA_2_FLAIR3D.to(device)
            self.LIDARHD_2_FLAIR3D_on_device = self.LIDARHD_2_FLAIR3D.to(device)
            self.DISAGREEMENT_COLORS_on_device = self.DISAGREEMENT_COLORS.to(device)
            self.DISAGREEMENT_COLORS_CLASS_OF_INTEREST_on_device = \
                self.DISAGREEMENT_COLORS_CLASS_OF_INTEREST.to(device)
            
            self.mapped_to_device = True

    @classmethod
    def histogram_to_onehot(cls, histogram): # to put in utils
        """
        Convert a histogram of labels to a onehot vector.
        """
        # Ignore the last column, which is the void class
        if histogram.dim() == 1:
            # already as onehot
            return histogram
        
        num_classes = histogram.shape[1] - 1
        histogram = histogram[:,:-1]
        void_only_mask = histogram.sum(dim=1) == 0
        
        onehot = histogram.argmax(dim=1)
        onehot[void_only_mask] = num_classes # void label
        
        return onehot