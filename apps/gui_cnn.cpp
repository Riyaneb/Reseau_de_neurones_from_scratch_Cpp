#include "utils/DatasetImage.hpp"
#include "utils/PredictionCnn.hpp"
#include <sstream>

#define Matrix RlMatrix
#include <iomanip>
#include <raylib.h>
#undef Matrix




Answer_gui prediction_image(DatasetImage& dataset, Index num_image, Shape const &s, std::string const &path, Index dataset_type);


int main() {
    Shape s(1,28,28);
    std::string path = "MNIST/test";
    Index type = 1;
    Index choice = 67;
    Index temp = choice;
    DatasetImage dataset_test(s,path,type);
    Answer_gui answer(prediction_image(dataset_test,choice, s, path, type));
    unsigned char r, g, b = 0;


    InitWindow(800,500, "Prédiction sur le dataset MNIST");
    Font font = LoadFont("data/font/Inter_18pt-Regular.ttf");
    SetTargetFPS(60);
    while (!WindowShouldClose()) {

        if (IsKeyPressed(KEY_RIGHT)) {
            if (choice < dataset_test.get_nImages() - 1) {
                choice++;
                answer = prediction_image(dataset_test,choice, s, path, type);
            }
        }

        if (IsKeyPressed(KEY_LEFT)) {
            if (choice > 0) {
                choice --;
                answer = prediction_image(dataset_test,choice, s, path, type);
            }
        }

        BeginDrawing();
        ClearBackground({40,40,40,255});
        DrawRectangle(28,28,284,284,BLACK);
        DrawRectangleLines(28,28,284,284,GRAY);
        for (Index h = 0; h < s.get_height(); h++) {
            for (Index w = 0; w < s.get_width(); w++) {
                if (s.get_channel() == 1) {
                    r = static_cast<unsigned char>(dataset_test.get_images().get_value(choice, s.get_column(0, h, w)) * 255.0f);
                    g = r;
                    b = r;
                }
                else if (s.get_channel() == 3) {
                    r = static_cast<unsigned char>(dataset_test.get_images().get_value(choice, s.get_column(0, h, w)) * 255.0f);
                    g = static_cast<unsigned char>(dataset_test.get_images().get_value(choice, s.get_column(1, h, w)) * 255.0f);;
                    b = static_cast<unsigned char>(dataset_test.get_images().get_value(choice, s.get_column(2, h, w)) * 255.0f);;
                }
                Color color = {r,g,b,255};
                DrawRectangle(30 + w * 10,30 + h * 10 ,10,10,color);
            }
        }
        std::ostringstream txt;
        txt << "Image " << choice;
        DrawTextEx(font,txt.str().c_str(),{115,330},25,2,WHITE);
        DrawTextEx(font, "Utilisez les fleches directionnel gauche et droite pour changer d'image",{ 28, 450}, 18, 2,WHITE);
        txt.str("");
        txt.clear();
        txt << "Le reseau devine \"" << answer.answer_title << "\" a " << std::fixed << std::setprecision(2) << answer.answer_score << "%";

        DrawTextEx(font, txt.str().c_str(),{340,42},25,2,WHITE);

        for (Index i = 0; i < dataset_test.get_nClass(); i++) {
            txt.str("");
            txt.clear();
            txt << answer.class_names[i];
            DrawTextEx(font, txt.str().c_str(),{static_cast<float>(380 - MeasureTextEx(font,txt.str().c_str(), 15,2).x),static_cast<float>(100) + i * 25},15,2,WHITE);
            DrawRectangle(400, 100 + i * 25, 320, 20, BLACK);
            if (i != answer.answer_index) {
                DrawRectangle(400, 100 + i * 25, 320 * answer.score[i] , 20, GRAY);
            }
            else if (answer.answer) {
                DrawRectangle(400, 100 + i * 25, 320 * answer.score[i] , 20, GREEN);
            }
            else {
                DrawRectangle(400, 100 + i * 25, 320 * answer.score[i] , 20, RED);
            }
            txt.str("");
            txt.clear();
            txt << std::fixed << std::setprecision(2) << answer.score[i] * 100 << "%";
            DrawTextEx(font, txt.str().c_str(), {730, static_cast<float>(100) + i * 25}, 18, 2,WHITE);
        }

        txt.str("");
        txt.clear();
        txt << "Vrai resultat : " << answer.real_result;
        if (answer.answer) {
            txt << ". Reponse correcte";
        }
        else {
            txt << ". Reponse incorrecte";
        }
        DrawTextEx(font, txt.str().c_str(), {400,370}, 18, 2,WHITE);
        EndDrawing();
    }
    UnloadFont(font);
    CloseWindow();

    return 0;
}
