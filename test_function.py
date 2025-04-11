import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compare_boards_and_num_pieces(gt_data, pred_data):
    results = []

    for gt, pred in zip(gt_data, pred_data):
        image_name = gt['image']
        board_gt = gt['board']
        board_pred = pred['board']
        num_gt = gt['num_pieces']
        num_pred = pred['num_pieces']

        # Compare number of pieces
        pieces_diff = abs(num_gt - num_pred)

        # Compare piece positions (cell by cell)
        total_cells = 64
        matches = 0
        for i in range(8):
            for j in range(8):
                if board_gt[i][j] == board_pred[i][j]:
                    matches += 1

        accuracy_percentage = (matches / total_cells) * 100

        results.append({
            "image": image_name,
            "num_pieces_gt": num_gt,
            "num_pieces_pred": num_pred,
            "Total failed pieces": pieces_diff,
            "matches_in_64": matches,
            "board_accuracy_percentage": round(accuracy_percentage, 2)
        })

    return results


gt = load_json('output_test.json')       # file with the correct board and num_pieces
pred = load_json('output_test.json')     # file with the generated results

res = compare_boards_and_num_pieces(gt, pred)

for r in res:
    print(r)
