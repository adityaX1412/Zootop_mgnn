import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Node:
    id: int
    longitude: float
    latitude: float

@dataclass
class Link:
    id: int
    source: int
    target: int
    pre_installed_capacity: float
    pre_installed_cost: float
    routing_cost: float
    setup_cost: float
    module_capacity: float
    module_cost: float

@dataclass
class Demand:
    id: int
    source: int
    target: int
    routing_unit: int
    demand_value: float
    max_path_length: str

class SNDLibParser:
    def __init__(self, content: str):
        self.content = content
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[int, Link] = {}
        self.demands: Dict[int, Demand] = {}
        
    def parse(self):
        # Split into sections
        sections = self.content.split('#')
        
        # Parse nodes
        nodes_section = self._find_section(sections, "NODE SECTION")
        if nodes_section:
            self._parse_nodes(nodes_section)
            
        # Parse links
        links_section = self._find_section(sections, "LINKS")
        if links_section:
            self._parse_links(links_section)
            
        # Parse demands
        demands_section = self._find_section(sections, "DEMAND SECTION")
        if demands_section:
            self._parse_demands(demands_section)
            
    def _find_section(self, sections: List[str], section_name: str) -> str:
        for section in sections:
            if section_name in section:
                return section
        return ""
    
    def _parse_nodes(self, section: str):
        pattern = r'(\d+)\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)'
        matches = re.finditer(pattern, section)
        
        for match in matches:
            node_id = int(match.group(1))
            longitude = float(match.group(2))
            latitude = float(match.group(3))
            self.nodes[node_id] = Node(node_id, longitude, latitude)
    
    def _parse_links(self, section: str):
        pattern = r'(\d+)\s*\(\s*(\d+)\s+(\d+)\s*\)\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\(\s*([\d.]+)\s+([\d.]+)\s*\)'
        matches = re.finditer(pattern, section)
        
        for match in matches:
            link_id = int(match.group(1))
            self.links[link_id] = Link(
                id=link_id,
                source=int(match.group(2)),
                target=int(match.group(3)),
                pre_installed_capacity=float(match.group(4)),
                pre_installed_cost=float(match.group(5)),
                routing_cost=float(match.group(6)),
                setup_cost=float(match.group(7)),
                module_capacity=float(match.group(8)),
                module_cost=float(match.group(9))
            )
    
    def _parse_demands(self, section: str):
        pattern = r'(\d+)\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\d+)\s+([\d.]+)\s+(\w+)'
        matches = re.finditer(pattern, section)
        
        for match in matches:
            demand_id = int(match.group(1))
            self.demands[demand_id] = Demand(
                id=demand_id,
                source=int(match.group(2)),
                target=int(match.group(3)),
                routing_unit=int(match.group(4)),
                demand_value=float(match.group(5)),
                max_path_length=match.group(6)
            )

# Example usage
def load_network(file_content: str):
    parser = SNDLibParser(file_content)
    parser.parse()
    return parser.nodes, parser.links, parser.demands
