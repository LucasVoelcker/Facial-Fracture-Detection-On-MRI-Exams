# CÓDIGO RODADO NO GOOGLE COLAB, DENTRO DO GOOGLE DRIVE

import cv2
import os


# Função para pintar de preto (cor do fundo) a parte da imagem com nome do paciente e data do exame
def hide_patient_info(img_path):
  img = cv2.imread(img_path)

  if img is None:
      raise FileNotFoundError(f"Não foi possível carregar a imagem: {img_path}")

  height, width = img.shape[:2]

  first_line_y = int(0.05 * height)
  second_line_y = int(0.08 * height)
  second_line_x = int(0.15 * width)

  img[0:first_line_y, :] = (0, 0, 0)
  img[0:second_line_y, 0:second_line_x] = (0, 0, 0)

  return img


root = "/content/drive/MyDrive/2025/DoutoradoMauricio/imagens/Tomo"
out_root = "/content/drive/MyDrive/2025/DoutoradoMauricio/imagens/Anonimizadas-17-11-2025"

for year in sorted(os.listdir(root)):
  patient_id = 0
  year_path = os.path.join(root, year)
  if year == '2023':
    year_path = os.path.join(year_path, year)
  patients = os.listdir(year_path)

  for patient in sorted(patients):
    patient_str = str(patient)

    patient_parts = patient_str.rsplit("-", 1)
    first_part = patient_parts[0]

    patient_path = os.path.join(year_path, patient)
    patient_numbers = os.listdir(patient_path)
    patient_number = patient_numbers[0]
    patient_number_path = os.path.join(patient_path, patient_number)
    exams = os.listdir(patient_number_path)

    for exam in sorted(exams):
      exam_path = os.path.join(patient_number_path, exam)
      images = os.listdir(exam_path)

      for image in sorted(images):
        img_path = os.path.join(exam_path, image)
        out_img = hide_patient_info(img_path)

        # Anonimiza o nome da pasta dos pacientes
        patient_folder = f"{first_part} - Paciente {patient_id}"

        if year == "2023":
          out_path = os.path.join(out_root, year, year, patient_folder, patient_number, exam, image)
        else:
          out_path = os.path.join(out_root, year, patient_folder, patient_number, exam, image)
        try:
          os.makedirs(os.path.dirname(out_path), exist_ok=True)
          cv2.imwrite(out_path, out_img)
        except Exception as e:
          print(f"Erro ao salvar imagem: {e}")

    patient_id = patient_id + 1