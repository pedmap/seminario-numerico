import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ajuste_linear(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Realiza o ajuste linear aos dados.

    Parâmetros:
    ----------
    x : numpy.ndarray
        Valores da variável independente.
    y : numpy.ndarray
        Valores da variável dependente.

    Retorna:
    --------
    tuple
        Previsões, intercepto, coeficiente angular e R².
    """

    # Calcula o número de pontos
    n = len(x)

    # Calcula as somas necessárias para a fórmula de regressão linear
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_y = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    # Calcula o coeficiente angular (b_linear) e o intercepto (a_linear)
    b_linear = (n * sum_x_y - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    a_linear = (sum_y - b_linear * sum_x) / n

    # Gera as previsões da equação linear y = a + bx
    y_linear_pred = a_linear + b_linear * x

    # Calcula o R² para o ajuste linear 
    r_sqr_linear = calcular_r_sqr(y, y_linear_pred)

    return y_linear_pred, a_linear, b_linear, r_sqr_linear

def ajuste_exponencial(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Realiza o ajuste exponencial aos dados.

    Parâmetros:
    ----------
    x : numpy.ndarray
        Valores da variável independente.
    y : numpy.ndarray
        Valores da variável dependente (devem ser positivos).

    Retorna:
    --------
    tuple
        Previsões, coeficiente de escala, taxa de crescimento e R².

    Levanta:
    -------
    ValueError se y contiver valores menores que zero.
    """
    # Troca de variável y para linearizar a equação exponencial
    log_y = np.log(y)

    # Calcula as somas necessárias para o ajuste exponencial
    n_exp = len(log_y)
    sum_log_y = np.sum(log_y)
    sum_x_exp = np.sum(x)
    sum_x_log_y = np.sum(x * log_y)
    sum_x2_exp = np.sum(x ** 2)

    # Calcula os coeficientes da equação exponencial 
    b_exp = (n_exp * sum_x_log_y - sum_x_exp * sum_log_y) / (n_exp * sum_x2_exp - sum_x_exp ** 2)
    ln_a_exp = (sum_log_y - b_exp * sum_x_exp) / n_exp
    a_exp = np.exp(ln_a_exp)  # a é obtido exponenciando ln_a

    # Gera as previsões da equação y = a * exp(b * x)
    y_exp_pred = a_exp * np.exp(b_exp * x)

    # Calcula o R² para o ajuste exponencial
    r_sqr_exp = calcular_r_sqr(y, y_exp_pred)

    return y_exp_pred, a_exp, b_exp, r_sqr_exp


def ajuste_logaritmico(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Realiza o ajuste logarítmico aos dados.

    Parâmetros:
    ----------
    x : numpy.ndarray
        Valores da variável independente (devem ser positivos).
    y : numpy.ndarray
        Valores da variável dependente.

    Retorna:
    --------
    tuple
        Previsões, intercepto, coeficiente angular e R².

    Levanta:
    -------
    ValueError se x contiver valores menores ou iguais a zero.
    """
    # Troca de variável x para linearizar a equação logarítmica
    log_x = np.log(x)

    # Calcula as somas necessárias para o ajuste logarítmico
    n_log = len(log_x)
    sum_log_x = np.sum(log_x)
    sum_y_log = np.sum(y)
    sum_log_x_y = np.sum(log_x * y)
    sum_log_x2 = np.sum(log_x ** 2)

    # Calcula os coeficientes da equação logarítmica y = a + b * log(x)
    b_log = (n_log * sum_log_x_y - sum_log_x * sum_y_log) / (n_log * sum_log_x2 - sum_log_x ** 2)
    a_log = (sum_y_log - b_log * sum_log_x) / n_log

    # Gera as previsões com base no modelo logarítmico ajustado
    y_log_pred = a_log + b_log * log_x

    # Calcula o R² para o ajuste logarítmico
    r_sqr_log = calcular_r_sqr(y, y_log_pred)

    return y_log_pred, a_log, b_log, r_sqr_log

def ajuste_polinomial(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Realiza o ajuste polinomial de segundo grau aos dados.

    Parâmetros:
    ----------
    x : numpy.ndarray
        Valores da variável independente.
    y : numpy.ndarray
        Valores da variável dependente.

    Retorna:
    --------
    tuple
        Previsões, coeficientes do polinômio e R².
    """
    # Calcula os coeficientes do polinômio de segundo grau (ax^2 + bx + c)
    coefs_poli = np.polyfit(x, y, 2)
    a_poli, b_poli, c_poli = coefs_poli

    # Gera as previsões com base no modelo polinomial ajustado
    y_poli_pred = a_poli * x**2 + b_poli * x + c_poli

    # Calcula o R² para o ajuste polinomial
    r_sqr_poli = calcular_r_sqr(y, y_poli_pred)

    return y_poli_pred, a_poli, b_poli, c_poli, r_sqr_poli

def calcular_r_sqr(y: np.ndarray, pred: np.ndarray) -> float:
    """
    Calcula o coeficiente de determinação R².

    Parâmetros:
    ----------
    y : numpy.ndarray
        Valores da variável dependente.
    pred : numpy.ndarray
        Valores previstos pelo modelo.

    Retorna:
    --------
    float
        Coeficiente de determinação R².
    """
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - pred) ** 2)
    r_sqr = 1 - (ss_residual / ss_total)
    return r_sqr

def plot_ajustes(x, y, y_linear_pred, a_linear, b_linear, r_sqr_linear,
                 y_exp_pred, a_exp, b_exp, r_sqr_exp,
                 y_log_pred, a_log, b_log, r_sqr_log,
                 y_poli_pred, a_poli, b_poli, c_poli, r_sqr_poli):
    """Plota os resultados dos ajustes em uma grade 2x2."""
    plt.figure(figsize=(14, 12))

    texto_plots = {
        "linear": f'Ajuste Linear:\n$y = {a_linear:.2f} + {b_linear:.2f} \cdot x$\n$R^2 = {r_sqr_linear:.4f}$',
        "exponencial": f'Ajuste Exponencial:\n$y = {a_exp:.2e} \cdot e^{{{b_exp:.4e} \cdot x}}$\n$R^2 = {r_sqr_exp:.4f}$',
        "logaritmo": f'Ajuste Logarítmico:\n$y = {a_log:.2f} + {b_log:.2f} \cdot \log(x)$\n$R^2 = {r_sqr_log:.4f}$',
        "polinomial":  f'Ajuste Polinomial:\n$y ={a_poli:.2f} \cdot x^2 {"+" if b_poli > 0 else ""} '
            f'{b_poli:.2f} \cdot x + {c_poli:.2f}$\n$R^2 = {r_sqr_poli:.4f}$'
    }

    # Plot do ajuste linear
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, label='Dados reais', color='darkblue', marker='o', edgecolor='white', s=60)
    plt.plot(x, y_linear_pred, color='orange', linestyle='-', linewidth=2, label='Ajuste Linear')
    plt.text(0.05, 0.95,
        texto_plots["linear"],
        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Ajuste Linear', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Variação (mm)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot do ajuste exponencial
    plt.subplot(2, 2, 2)
    plt.scatter(x, y, label='Dados reais', color='darkblue', marker='o', edgecolor='white', s=60)
    plt.plot(x, y_exp_pred, color='magenta', linestyle='-', linewidth=2, label='Ajuste Exponencial')
    plt.text(0.05, 0.95,
        texto_plots["exponencial"],
        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Ajuste Exponencial', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Variação (mm)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot do ajuste logarítmico
    plt.subplot(2, 2, 3)
    plt.scatter(x, y, label='Dados reais', color='darkblue', marker='o', edgecolor='white', s=60)
    plt.plot(x, y_log_pred, color='green', linestyle='-', linewidth=2, label='Ajuste Logarítmico')
    plt.text(0.05, 0.95,
        texto_plots["logaritmo"],
        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Ajuste Logarítmico', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Variação (mm)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot do ajuste polinomial
    plt.subplot(2, 2, 4)
    plt.scatter(x, y, label='Dados reais', color='darkblue', marker='o', edgecolor='white', s=60)
    plt.plot(x, y_poli_pred, color='red', linestyle='-', linewidth=2, label='Ajuste Polinomial')
    plt.text(0.05, 0.95,
       texto_plots["polinomial"],
        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Ajuste Polinomial', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Variação (mm)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Ajustar o layout para evitar sobreposição de textos
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()

def verificar_negativos(dados):
    # Verificar se há valores de y menores ou iguais a zero
    if np.any(dados <= 0):
        raise ValueError("Os valores de y devem ser positivos para um ajuste exponencial.")

def exibir_resultados(a_linear, b_linear, r_sqr_linear,
                     a_exp, b_exp, r_sqr_exp,
                     a_log, b_log, r_sqr_log,
                     a_poli, b_poli, c_poli, r_sqr_poli):
    resultados = {
        "Modelo": ["Linear", "Exponencial", "Logarítmico", "Polinomial"],
        "a": [a_linear, a_exp, a_log, a_poli],
        "b": [b_linear, b_exp, b_log, b_poli],
        "c": ["-", "-", "-", c_poli],
        "R²": [r_sqr_linear, r_sqr_exp, r_sqr_log, r_sqr_poli]
    }
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados)

def main():
    """Função principal para carregar dados, realizar ajustes e plotar resultados."""

    data = pd.read_csv('serie.csv', sep=';', decimal=',')

    x = data['ano'].values                # Anos
    y = data['variacao(mm)'].values       # Variações

    verificar_negativos(y)

    # Realizar os ajustes
    y_linear_pred, a_linear, b_linear, r_sqr_linear = ajuste_linear(x, y)
    y_exp_pred, a_exp, b_exp, r_sqr_exp = ajuste_exponencial(x, y)
    y_log_pred, a_log, b_log, r_sqr_log = ajuste_logaritmico(x, y)
    y_poli_pred, a_poli, b_poli, c_poli, r_sqr_poli = ajuste_polinomial(x, y)
    
    # Projeções para os anos 2050, 2075 e 2100
    anos_projecao = np.array([2050, 2075, 2100])
    projecao_linear = a_linear + b_linear * anos_projecao
    projecao_exponencial = a_exp * np.exp(b_exp * anos_projecao)
    projecao_logaritmica = a_log + b_log * np.log(anos_projecao)
    projecao_polinomial = a_poli * anos_projecao**2 + b_poli * anos_projecao + c_poli

    # Exibir os resultados das projeções de maneira mais legível
    print("\nProjeções para os anos 2050, 2075 e 2100:")
    print(f"{'Ano':<8}{'Linear (mm)':<15}{'Exponencial (mm)':<20}{'Logarítmico (mm)':<20}{'Polinomial (mm)':<20}")
    print("-" * 75)
    for ano, lin, exp, log, poli in zip(anos_projecao, projecao_linear, projecao_exponencial, projecao_logaritmica, projecao_polinomial):
        print(f"{ano:<8}{lin:<15.2f}{exp:<20.2f}{log:<20.2f}{poli:<20.2f}")

    # Plotar os resultados
    plot_ajustes(x, y, y_linear_pred, a_linear, b_linear, r_sqr_linear,
                 y_exp_pred, a_exp, b_exp, r_sqr_exp,
                 y_log_pred, a_log, b_log, r_sqr_log,
                 y_poli_pred, a_poli, b_poli, c_poli, r_sqr_poli)

    exibir_resultados(a_linear, b_linear, r_sqr_linear, 
                    a_exp, b_exp, r_sqr_exp,
                    a_log, b_log, r_sqr_log,
                    a_poli, b_poli, c_poli, r_sqr_poli)

if __name__ == "__main__":
    main()
