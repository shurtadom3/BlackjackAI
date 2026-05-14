import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_metrics(csv_path='data/metrics/training_metrics.csv', save_path='data/metrics/win_rate_plot.png'):
    """
    Carga las métricas de entrenamiento y grafica la evolución del win rate.
    """
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo de métricas en {csv_path}")
        return

    # Cargar las métricas de entrenamiento
    metrics = pd.read_csv(csv_path)

    # Graficar win rate
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['win_rate'], label='Win Rate acumulado', color='b')
    plt.xlabel('Episodio')
    plt.ylabel('Win Rate')
    plt.title('Evolución del Win Rate durante el entrenamiento')
    plt.grid(True)
    plt.legend()
    
    # Guardamos la imagen para asegurarnos de que la gráfica se genera correctamente
    plt.savefig(save_path)
    print(f"Gráfica guardada exitosamente en: {save_path}")
    
    # Mostramos la gráfica en pantalla
    plt.show(block=False)
    plt.pause(3) # Pausamos 3 segundos para que puedas verla antes de que el script termine si se ejecuta en consola
    plt.close()

if __name__ == "__main__":
    plot_training_metrics()
