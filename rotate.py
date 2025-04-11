def encontrar_canto_mais_escuro(imagem, margem=100):
    h, w = imagem.shape[:2]

    # Extrair os 4 cantos da imagem
    cantos = {
        'top_left': imagem[0:margem, 0:margem],
        'top_right': imagem[0:margem, w-margem:w],
        'bottom_right': imagem[h-margem:h, w-margem:w],
        'bottom_left': imagem[h-margem:h, 0:margem]
    }

    # Calcular a média da intensidade em escala de cinza
    medias = {
        nome: np.mean(cv2.cvtColor(canto, cv2.COLOR_BGR2GRAY))
        for nome, canto in cantos.items()
    }

    # Retornar o canto mais escuro
    return min(medias, key=medias.get)

def rotacionar_para_inferior_esquerdo(imagem, canto_origem):
    if canto_origem == 'top_left':
        # 90° anti-horário
        return cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif canto_origem == 'top_right':
        # 180°
        return cv2.rotate(imagem, cv2.ROTATE_180)
    elif canto_origem == 'bottom_right':
        # 90° horário
        return cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    else:
        # Já está no canto inferior esquerdo
        return imagem


canto_logo = encontrar_canto_mais_escuro(warped)
imagem_corrigida = rotacionar_para_inferior_esquerdo(warped, canto_logo)