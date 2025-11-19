import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import copy

class GraphAlgorithmsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmos de Grafos - Fleury, Hierholzer y Kaufmann-Malgrange")
        self.root.geometry("1200x800")
        
        # Variables
        self.graph = None
        self.adj_matrix = None
        self.matrix_entries = []
        
        self.setup_ui()
        
        # Cerrar matplotlib correctamente al salir
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Manejar el cierre de la ventana"""
        try:
            plt.close('all')
        except:
            pass
        self.root.destroy()
    
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar peso de filas y columnas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Frame izquierdo - Entrada y controles
        self.setup_left_panel(main_frame)
        
        # Frame derecho superior - Visualización
        self.setup_visualization_panel(main_frame)
        
        # Frame derecho inferior - Consola
        self.setup_console_panel(main_frame)
    
    def setup_left_panel(self, parent):
        """Configurar panel izquierdo con controles"""
        left_frame = ttk.Frame(parent, padding="5")
        left_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)
        
        # Título
        ttk.Label(left_frame, text="Matriz de Adyacencia:", 
                  font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5, sticky=tk.W)
        
        # Control de tamaño
        size_frame = ttk.Frame(left_frame)
        size_frame.grid(row=1, column=0, pady=5, sticky=tk.W)
        
        ttk.Label(size_frame, text="Tamaño:").pack(side=tk.LEFT, padx=5)
        self.matrix_size = tk.IntVar(value=5)
        ttk.Spinbox(size_frame, from_=2, to=10, textvariable=self.matrix_size, 
                   width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(size_frame, text="Crear Matriz", 
                  command=self.create_matrix_grid).pack(side=tk.LEFT, padx=5)
        
        # Contenedor de matriz con scroll
        matrix_container = ttk.Frame(left_frame)
        matrix_container.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        matrix_container.columnconfigure(0, weight=1)
        matrix_container.rowconfigure(0, weight=1)
        
        # Canvas con scrollbars
        self.matrix_canvas = tk.Canvas(matrix_container, width=380, height=300, bg='white')
        scrollbar_y = ttk.Scrollbar(matrix_container, orient="vertical", 
                                    command=self.matrix_canvas.yview)
        scrollbar_x = ttk.Scrollbar(matrix_container, orient="horizontal", 
                                    command=self.matrix_canvas.xview)
        
        self.matrix_frame = ttk.Frame(self.matrix_canvas)
        self.matrix_canvas.create_window((0, 0), window=self.matrix_frame, anchor="nw")
        self.matrix_canvas.configure(yscrollcommand=scrollbar_y.set, 
                                     xscrollcommand=scrollbar_x.set)
        
        self.matrix_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Actualizar región de scroll al cambiar tamaño
        self.matrix_frame.bind('<Configure>', 
                              lambda e: self.matrix_canvas.configure(
                                  scrollregion=self.matrix_canvas.bbox("all")))
        
        # Crear matriz inicial
        self.create_matrix_grid()
        
        # Botón cargar
        ttk.Button(left_frame, text="Cargar Matriz y Visualizar Grafo", 
                   command=self.load_matrix).grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Botones de algoritmos
        self.setup_algorithm_buttons(left_frame)
    
    def setup_algorithm_buttons(self, parent):
        """Configurar botones de algoritmos"""
        ttk.Label(parent, text="Algoritmos:", 
                  font=('Arial', 12, 'bold')).grid(row=4, column=0, pady=10, sticky=tk.W)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=5, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(btn_frame, text="Algoritmo de Fleury", 
                   command=self.run_fleury, width=30).pack(pady=5, fill=tk.X)
        ttk.Button(btn_frame, text="Algoritmo de Hierholzer", 
                   command=self.run_hierholzer, width=30).pack(pady=5, fill=tk.X)
        ttk.Button(btn_frame, text="Algoritmo de Kaufmann-Malgrange", 
                   command=self.run_kaufmann_malgrange, width=30).pack(pady=5, fill=tk.X)
        
        ttk.Separator(btn_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Limpiar Consola", 
                   command=self.clear_console, width=30).pack(pady=5, fill=tk.X)
    
    def setup_visualization_panel(self, parent):
        """Configurar panel de visualización"""
        vis_frame = ttk.LabelFrame(parent, text="Visualización del Grafo", padding="5")
        vis_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial
        self.ax.text(0.5, 0.5, 'Cargue una matriz para visualizar el grafo', 
                    ha='center', va='center', fontsize=12, color='gray')
        self.ax.axis('off')
        self.canvas.draw()
    
    def setup_console_panel(self, parent):
        """Configurar panel de consola"""
        console_frame = ttk.LabelFrame(parent, text="Consola de Salida", padding="5")
        console_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, width=60, height=15, 
                                                  font=('Courier', 10), bg='#1e1e1e', 
                                                  fg='#00ff00', insertbackground='white')
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Mensaje de bienvenida
        self.log("=" * 50)
        self.log("Sistema de Análisis de Grafos")
        self.log("=" * 50)
        self.log("Cargue una matriz de adyacencia y ejecute un algoritmo.")
        self.log("")
    
    def create_matrix_grid(self):
        """Crear cuadrícula editable de la matriz"""
        # Limpiar matriz anterior
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        
        self.matrix_entries = []
        n = self.matrix_size.get()
        
        # Validación de entrada numérica
        vcmd = (self.root.register(self.validate_number), '%P')
        
        # Encabezado de columnas
        ttk.Label(self.matrix_frame, text="", width=3).grid(row=0, column=0, padx=2, pady=2)
        for j in range(n):
            ttk.Label(self.matrix_frame, text=str(j), width=5, 
                     font=('Arial', 10, 'bold'), 
                     anchor='center').grid(row=0, column=j+1, padx=2, pady=2)
        
        # Matriz de ejemplo
        default_matrix = [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0]
        ]
        
        # Crear celdas
        for i in range(n):
            # Encabezado de fila
            ttk.Label(self.matrix_frame, text=str(i), width=3, 
                     font=('Arial', 10, 'bold'), 
                     anchor='center').grid(row=i+1, column=0, padx=2, pady=2)
            
            row_entries = []
            for j in range(n):
                entry = ttk.Entry(self.matrix_frame, width=5, justify='center',
                                 validate='key', validatecommand=vcmd)
                entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                
                # Valor por defecto
                if i < len(default_matrix) and j < len(default_matrix[0]):
                    entry.insert(0, str(default_matrix[i][j]))
                else:
                    entry.insert(0, "0")
                
                row_entries.append(entry)
            
            self.matrix_entries.append(row_entries)
        
        # Actualizar región de scroll
        self.matrix_frame.update_idletasks()
        self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all"))
    
    def validate_number(self, value):
        """Validar entrada numérica"""
        if value == "":
            return True
        try:
            int(value)
            return True
        except ValueError:
            return False
    
    def log(self, message):
        """Escribir en la consola"""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.root.update_idletasks()
    
    def clear_console(self):
        """Limpiar consola"""
        self.console.delete('1.0', tk.END)
    
    def load_matrix(self):
        """Cargar matriz desde la cuadrícula"""
        try:
            n = self.matrix_size.get()
            self.adj_matrix = []
            
            # Leer valores
            for i in range(n):
                row = []
                for j in range(n):
                    value = self.matrix_entries[i][j].get().strip()
                    if value == "":
                        value = "0"
                    row.append(int(value))
                self.adj_matrix.append(row)
            
            # Validar simetría (grafo no dirigido)
            for i in range(n):
                for j in range(i+1, n):
                    if self.adj_matrix[i][j] != self.adj_matrix[j][i]:
                        raise ValueError(f"La matriz no es simétrica en posición ({i},{j})")
            
            # Crear grafo
            self.graph = nx.Graph()
            for i in range(n):
                self.graph.add_node(i)
                for j in range(i+1, n):
                    if self.adj_matrix[i][j] > 0:
                        self.graph.add_edge(i, j, weight=self.adj_matrix[i][j])
            
            # Verificar que el grafo no esté vacío
            if self.graph.number_of_edges() == 0:
                raise ValueError("El grafo no tiene aristas")
            
            self.visualize_graph()
            self.log("\n" + "=" * 50)
            self.log("✓ Matriz cargada exitosamente")
            self.log(f"  Nodos: {n}")
            self.log(f"  Aristas: {self.graph.number_of_edges()}")
            self.log(f"  Conexo: {'Sí' if nx.is_connected(self.graph) else 'No'}")
            
            # Mostrar grados
            degrees = dict(self.graph.degree())
            self.log(f"\n  Grados de nodos: {degrees}")
            self.log("=" * 50)
            
        except ValueError as e:
            messagebox.showerror("Error de validación", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la matriz: {str(e)}")
    
    def visualize_graph(self):
        """Visualizar el grafo"""
        self.ax.clear()
        
        try:
            # Layout para mejor visualización
            if len(self.graph.nodes()) <= 6:
                pos = nx.circular_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph, seed=42, k=1, iterations=50)
            
            # Dibujar nodos
            nx.draw_networkx_nodes(self.graph, pos, node_color='#4CAF50', 
                                   node_size=700, alpha=0.9, ax=self.ax)
            
            # Dibujar aristas
            nx.draw_networkx_edges(self.graph, pos, width=2.5, alpha=0.6, 
                                   edge_color='#2196F3', ax=self.ax)
            
            # Etiquetas de nodos
            nx.draw_networkx_labels(self.graph, pos, font_size=14, 
                                    font_weight='bold', font_color='white', ax=self.ax)
            
            # Pesos de aristas (si existen y no son todos 1)
            edge_labels = nx.get_edge_attributes(self.graph, 'weight')
            if edge_labels and not all(w == 1 for w in edge_labels.values()):
                nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, 
                                            font_size=10, ax=self.ax)
            
            self.ax.set_title(f"Grafo con {self.graph.number_of_nodes()} nodos y "
                            f"{self.graph.number_of_edges()} aristas", 
                            fontsize=12, fontweight='bold', pad=20)
            self.ax.axis('off')
            self.ax.margins(0.1)
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f'Error al visualizar: {str(e)}', 
                        ha='center', va='center', color='red')
            self.ax.axis('off')
        
        self.canvas.draw()
    
    # ==================== ALGORITMO DE FLEURY ====================
    
    def run_fleury(self):
        """Ejecutar algoritmo de Fleury"""
        if not self.validate_graph():
            return
        
        self.log("\n" + "=" * 50)
        self.log("ALGORITMO DE FLEURY")
        self.log("=" * 50)
        
        # Análisis del grafo
        if not nx.is_connected(self.graph):
            self.log("\n❌ ERROR: El grafo no es conexo")
            self.log("   No puede tener caminos/circuitos eulerianos")
            self.log("=" * 50)
            return
        
        odd_degree_nodes = [v for v in self.graph.nodes() if self.graph.degree(v) % 2 == 1]
        
        self.log("\nANÁLISIS DEL GRAFO:")
        self.log(f"  Nodos con grado impar: {len(odd_degree_nodes)}")
        if odd_degree_nodes:
            self.log(f"  Nodos impares: {odd_degree_nodes}")
        
        if len(odd_degree_nodes) > 2:
            self.log("\n❌ El grafo NO es euleriano")
            self.log("   (Más de 2 nodos con grado impar)")
            self.log("=" * 50)
            return
        
        if len(odd_degree_nodes) == 0:
            self.log("\n✓ EULERIANO COMPLETO (Circuito Euleriano)")
            self.log("  • Todos los nodos tienen grado par")
            self.log("  • El camino empieza y termina en el mismo nodo")
            start_node = list(self.graph.nodes())[0]
        else:
            self.log("\n✓ SEMI-EULERIANO (Camino Euleriano)")
            self.log("  • Exactamente 2 nodos con grado impar")
            self.log("  • El camino empieza en un nodo impar y termina en el otro")
            start_node = odd_degree_nodes[0]
        
        # Ejecutar algoritmo
        g_copy = self.graph.copy()
        path = self.fleury_algorithm(g_copy, start_node)
        
        if path:
            self.log(f"\nRECORRIDO ENCONTRADO:")
            self.log("  " + " → ".join(map(str, path)))
            self.log(f"\n  Nodos visitados: {len(path)}")
            self.log(f"  Aristas recorridas: {len(path) - 1}")
        else:
            self.log("\n❌ No se pudo encontrar el recorrido")
        
        self.log("=" * 50)
    
    def fleury_algorithm(self, graph, start):
        """Implementación del algoritmo de Fleury"""
        if not graph or not graph.edges():
            return [start]
        
        path = [start]
        current = start
        
        while graph.edges():
            neighbors = list(graph.neighbors(current))
            
            if not neighbors:
                break
            
            # Buscar arista no puente
            next_node = None
            for neighbor in neighbors:
                graph.remove_edge(current, neighbor)
                
                if nx.is_connected(graph):
                    next_node = neighbor
                    break
                else:
                    graph.add_edge(current, neighbor)
            
            # Si todas son puentes, tomar cualquiera
            if next_node is None:
                next_node = neighbors[0]
                graph.remove_edge(current, next_node)
            
            path.append(next_node)
            current = next_node
        
        return path
    
    # ==================== ALGORITMO DE HIERHOLZER ====================
    
    def run_hierholzer(self):
        """Ejecutar algoritmo de Hierholzer"""
        if not self.validate_graph():
            return
        
        self.log("\n" + "=" * 50)
        self.log("ALGORITMO DE HIERHOLZER")
        self.log("=" * 50)
        
        # Verificar conectividad
        if not nx.is_connected(self.graph):
            self.log("\n❌ ERROR: El grafo no es conexo")
            self.log("=" * 50)
            return
        
        odd_degree_nodes = [v for v in self.graph.nodes() if self.graph.degree(v) % 2 == 1]
        
        self.log("\nANÁLISIS DEL GRAFO:")
        self.log(f"  Nodos con grado impar: {len(odd_degree_nodes)}")
        if odd_degree_nodes:
            self.log(f"  Nodos impares: {odd_degree_nodes}")
        
        if len(odd_degree_nodes) > 0:
            if len(odd_degree_nodes) == 2:
                self.log("\n❌ El grafo es SEMI-EULERIANO, no EULERIANO COMPLETO")
            else:
                self.log("\n❌ El grafo NO es euleriano")
            self.log("\nREQUISITO: Hierholzer solo funciona con CIRCUITOS EULERIANOS")
            self.log("           (Todos los nodos deben tener grado par)")
            self.log("\nSUGERENCIA: Use el algoritmo de Fleury para grafos semi-eulerianos")
            self.log("=" * 50)
            return
        
        self.log("\n✓ EULERIANO COMPLETO (Circuito Euleriano)")
        self.log("  • Todos los nodos tienen grado par")
        self.log("  • El circuito empieza y termina en el mismo nodo")
        
        # Ejecutar algoritmo
        start_node = list(self.graph.nodes())[0]
        path = self.hierholzer_algorithm(self.graph, start_node)
        
        if path:
            self.log(f"\nRECORRIDO ENCONTRADO:")
            self.log("  " + " → ".join(map(str, path)))
            self.log(f"\n  Nodos en el ciclo: {len(path)}")
            self.log(f"  Aristas recorridas: {len(path) - 1}")
        else:
            self.log("\n❌ No se pudo encontrar el circuito")
        
        self.log("=" * 50)
    
    def hierholzer_algorithm(self, graph, start):
        """Implementación del algoritmo de Hierholzer"""
        # Crear lista de adyacencia mutable
        adj = defaultdict(list)
        for u, v in graph.edges():
            adj[u].append(v)
            adj[v].append(u)
        
        stack = [start]
        path = []
        
        while stack:
            v = stack[-1]
            if adj[v]:
                u = adj[v].pop()
                # Remover arista en ambas direcciones
                if v in adj[u]:
                    adj[u].remove(v)
                stack.append(u)
            else:
                path.append(stack.pop())
        
        return path[::-1]
    
    # ==================== ALGORITMO DE KAUFMANN-MALGRANGE ====================
    
    def run_kaufmann_malgrange(self):
        """Ejecutar algoritmo de Kaufmann y Malgrange"""
        if not self.validate_graph():
            return
        
        self.log("\n" + "=" * 50)
        self.log("ALGORITMO DE KAUFMANN Y MALGRANGE")
        self.log("=" * 50)
        self.log("\nBuscando circuito hamiltoniano...")
        
        n = len(self.graph.nodes())
        
        # Buscar circuito hamiltoniano
        circuit = self.find_hamiltonian_circuit(self.graph, n)
        
        # Si no hay circuito, buscar camino
        path = None if circuit else self.find_hamiltonian_path(self.graph, n)
        
        self.log("\nANÁLISIS DEL GRAFO:")
        
        if circuit:
            self.log("\n✓ HAMILTONIANO COMPLETO (Circuito Hamiltoniano)")
            self.log("  • Visita todos los nodos exactamente una vez")
            self.log("  • El circuito regresa al nodo inicial")
            self.log("\nCIRCUITO ENCONTRADO:")
            self.log("  " + " → ".join(map(str, circuit)))
            self.log(f"\n  Nodos visitados: {len(circuit)}")
        elif path:
            self.log("\n✓ SEMI-HAMILTONIANO (Camino Hamiltoniano)")
            self.log("  • Visita todos los nodos exactamente una vez")
            self.log("  • NO regresa al nodo inicial")
            self.log("\nCAMINO ENCONTRADO:")
            self.log("  " + " → ".join(map(str, path)))
            self.log(f"\n  Nodos visitados: {len(path)}")
        else:
            self.log("\n❌ NO es hamiltoniano")
            self.log("  No existe un camino que visite todos los nodos")
            self.log("  exactamente una vez")
        
        self.log("=" * 50)
    
    def find_hamiltonian_circuit(self, graph, n):
        """Buscar circuito hamiltoniano (regresa al inicio)"""
        def is_safe(v, path, pos):
            if pos > 0 and not graph.has_edge(path[pos - 1], v):
                return False
            return v not in path[:pos]
        
        def hamiltonian_util(path, pos):
            if pos == n:
                return graph.has_edge(path[pos - 1], path[0])
            
            for v in sorted(graph.nodes()):
                if is_safe(v, path, pos):
                    path[pos] = v
                    if hamiltonian_util(path, pos + 1):
                        return True
                    path[pos] = -1
            
            return False
        
        for start in sorted(graph.nodes()):
            path = [-1] * (n + 1)
            path[0] = start
            
            if hamiltonian_util(path, 1):
                path[n] = path[0]
                return path
        
        return None
    
    def find_hamiltonian_path(self, graph, n):
        """Buscar camino hamiltoniano (sin regresar al inicio)"""
        def is_safe(v, path, pos):
            if pos > 0 and not graph.has_edge(path[pos - 1], v):
                return False
            return v not in path[:pos]
        
        def hamiltonian_util(path, pos):
            if pos == n:
                return True
            
            for v in sorted(graph.nodes()):
                if is_safe(v, path, pos):
                    path[pos] = v
                    if hamiltonian_util(path, pos + 1):
                        return True
                    path[pos] = -1
            
            return False
        
        for start in sorted(graph.nodes()):
            path = [-1] * n
            path[0] = start
            
            if hamiltonian_util(path, 1):
                return path
        
        return None
    
    def validate_graph(self):
        """Validar que el grafo esté cargado"""
        if self.graph is None:
            messagebox.showwarning("Advertencia", 
                                  "Primero cargue una matriz de adyacencia")
            return False
        return True


def main():
    """Función principal"""
    root = tk.Tk()
    app = GraphAlgorithmsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()