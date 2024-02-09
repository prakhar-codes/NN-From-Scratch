public class Layer {
    
    Node nodes[];
    Layer nextLayer;
    Layer prevLayer;

    public Layer() {
    }

    public void createNodes(int size) {
        nodes = new Node[size];
        for(int i=0; i<size; i++) {
            nodes[i] = new Node();
        }
    }
}
