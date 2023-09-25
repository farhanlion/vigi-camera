import xml.etree.ElementTree as ET
import pybgs as bgs
import cv2

class SuBSENSE:
    def __init__(self):
        self.config = './config/SuBSENSE.xml'
        self.bgs = bgs.SuBSENSE()

    def apply(self, frame):
        return self.bgs.apply(frame)
    
    def switchMode(self, mode):
        print('switching mode')
        tree = ET.parse('./config/SuBSENSE.xml')
        root = tree.getroot()
        thresh = root.findall('nMinColorDistThreshold')
        if mode == 'dark':
            thresh[0].text = '1'
        else:
            thresh[0].text = '30'
        tree.write('./config/SuBSENSE.xml',xml_declaration=True)