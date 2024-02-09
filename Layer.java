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

    public void fill(double data[]) {
        for(int i=0; i<this.nodes.length; i++) {
            this.nodes[i].value = data[i];
            this.nodes[i].activatedValue = data[i];
        }
    }

    public void fill() {
        for(int i=0; i<nodes.length; i++) {
            nodes[i].calculate(this.prevLayer);
        }
    }
}
