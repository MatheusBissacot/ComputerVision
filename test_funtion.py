import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def comparar_boards_e_num_pieces(gt_data, pred_data):
    resultados = []

    for gt, pred in zip(gt_data, pred_data):
        nome_imagem = gt['image']
        board_gt = gt['board']
        board_pred = pred['board']
        num_gt = gt['num_pieces']
        num_pred = pred['num_pieces']

        # Comparar número de peças
        pecas= abs(num_gt-num_pred)

        # Comparar posição das peças (celula a celula)
        total_celulas = 64
        acertos = 0
        for i in range(8):
            for j in range(8):
                if board_gt[i][j] == board_pred[i][j]:
                    acertos += 1

        porcentagem_acerto = (acertos / total_celulas) * 100

        resultados.append({
            "imagem": nome_imagem,
            "num_pieces_gt": num_gt,
            "num_pieces_pred": num_pred,
            "Total de peças falhadas": pecas,
            "acertos_em_64": acertos,
            "porcentagem_acerto_board": round(porcentagem_acerto, 2)
        })

    return resultados


gt = load_json('output_test.json')       # arquivo com o board e num_pieces corretos
pred = load_json('output_test.json')   # arquivo com os resultados gerados

res = comparar_boards_e_num_pieces(gt, pred)

for r in res:
    print(r)
