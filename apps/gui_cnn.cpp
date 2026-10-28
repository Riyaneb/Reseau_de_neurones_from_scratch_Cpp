#include "utils/DatasetImage.hpp"
#include "utils/PredictionCnn.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>

#define Matrix RlMatrix
#include <iomanip>
#include <raylib.h>
#undef Matrix

#define IMAGE_BOX 400
#define MARGIN 28
#define PAD 2
#define GAP 40
#define PANEL_PREDICTION 420
#define ROW_HIGH 25
#define FONT_SIZE_MEDIUM 25
#define FONT_SIZE_SMALL 18



Answer_gui prediction_image(DatasetImage& dataset, Index num_image, Shape const &s, std::string const &path, Index dataset_type);


int main() {
    Shape s(1,64,64);
    std::string path = "chest_xray/test";

    Index type = 0;
    Index choice = 67;

    DatasetImage dataset_test(s,path,type);
    dataset_test.shuffle();
    Answer_gui answer(prediction_image(dataset_test,choice, s, path, type));

    unsigned char r, g, b = 0;

    int size_pixel = std::max(1,IMAGE_BOX/ std::max(s.get_height(), s.get_width()));
    int size_image_width = size_pixel * s.get_width();
    int size_image_height = size_pixel * s.get_height();

    std::string biggest_percent = "100.00%";


    InitWindow(1000,800, "Prédiction sur le dataset chest xray");
    Font font = LoadFont("data/font/Inter_18pt-Regular.ttf");

    auto bigger_class_name = std::max_element(dataset_test.get_class_names().begin(), dataset_test.get_class_names().end(), [&font](const std::string& a, const std::string& b) {
        return MeasureTextEx(font, a.c_str(), FONT_SIZE_SMALL, 2).x < MeasureTextEx(font, b.c_str(), FONT_SIZE_SMALL, 2).x;
    });


    float win_x_right = MARGIN + size_image_width + GAP;
    float win_label_width = MeasureTextEx(font, bigger_class_name->c_str(), FONT_SIZE_SMALL, 2).x;
    float win_percent_width = MeasureTextEx(font, biggest_percent.c_str(), FONT_SIZE_SMALL, 2).x;
    float win_bars_end = win_x_right + win_label_width + 20 + PANEL_PREDICTION + GAP + win_percent_width;

    std::string longest_title = "Le reseau devine \"" + *bigger_class_name + "\" a 100.00%";
    std::string longest_verdict = "Vrai resultat : " + *bigger_class_name + ". Reponse incorrecte";
    Vector2 title_size = MeasureTextEx(font, longest_title.c_str(), FONT_SIZE_MEDIUM, 2);
    Vector2 verdict_size = MeasureTextEx(font, longest_verdict.c_str(), FONT_SIZE_SMALL, 2);

    std::string hint_text = "Utilisez les fleches directionnel gauche et droite pour changer d'image";
    Vector2 hint_size = MeasureTextEx(font, hint_text.c_str(), FONT_SIZE_SMALL, 2);
    float win_y_caption = MARGIN + size_image_height + MeasureTextEx(font, "Image", FONT_SIZE_MEDIUM, 2).y;
    float win_y_hint = win_y_caption + 75;

    float win_y_verdict = MARGIN + GAP + dataset_test.get_nClass() * GAP;

    int windows_width = static_cast<int>(std::max({win_bars_end, win_x_right + title_size.x, win_x_right + verdict_size.x, MARGIN + hint_size.x})) + 1 + MARGIN;
    int windows_height = static_cast<int>(std::max(win_y_hint + hint_size.y, win_y_verdict + verdict_size.y)) + 1 + MARGIN;

    SetWindowSize(windows_width, windows_height);
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


        DrawRectangle(MARGIN,MARGIN,size_image_width,size_image_height,BLACK);
        DrawRectangleLines(MARGIN,MARGIN,size_image_width + 2 * PAD, size_image_height + 2 * PAD, GRAY);

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
                else {
                    std::cerr << "Erreur : L'image " << choice << "doit posséder 1/3 canal/canaux de couleurs" << std::endl;
                }

                Color color = {r,g,b,255};

                DrawRectangle((MARGIN + PAD) + w * size_pixel,(MARGIN + PAD) + h * size_pixel ,size_pixel,size_pixel,color);
            }
        }
        std::ostringstream txt;
        txt << "Image " << choice;

        float x,y;
        x = MARGIN + (size_image_width / 2) - MeasureTextEx(font,txt.str().c_str(), FONT_SIZE_MEDIUM,2).x / 2;
        y = MARGIN + size_image_height + MeasureTextEx(font,txt.str().c_str(), FONT_SIZE_MEDIUM,2).y;

        DrawTextEx(font,txt.str().c_str(),{x ,y},FONT_SIZE_MEDIUM,2,WHITE);

        x = MARGIN;
        y = y + 75;
        DrawTextEx(font, "Utilisez les fleches directionnel gauche et droite pour changer d'image",{ x, y}, FONT_SIZE_SMALL, 2,WHITE);

        txt.str("");
        txt.clear();
        txt << "Le reseau devine \"" << answer.answer_title << "\" a " << std::fixed << std::setprecision(2) << answer.answer_score << "%";

        float x_right = MARGIN + size_image_width + GAP;
        y = MARGIN;

        DrawTextEx(font, txt.str().c_str(),{x_right,y},FONT_SIZE_MEDIUM,2,WHITE);

        for (Index i = 0; i < dataset_test.get_nClass(); i++) {
            txt.str("");
            txt.clear();
            txt << answer.class_names[i];

            x = x_right;
            y = MARGIN + GAP + i*GAP;

            DrawTextEx(font, txt.str().c_str(),{x, y},FONT_SIZE_SMALL,2,WHITE);

            x = x_right + win_label_width + 20;

            DrawRectangle(x, y, PANEL_PREDICTION, ROW_HIGH, BLACK);

            if (i != answer.answer_index) {
                DrawRectangle(x, y, PANEL_PREDICTION * answer.score[i] , 20, GRAY);
            }
            else if (answer.answer) {
                DrawRectangle(x, y, PANEL_PREDICTION * answer.score[i] , 20, GREEN);
            }
            else {
                DrawRectangle(x, y, PANEL_PREDICTION * answer.score[i] , 20, RED);
            }

            txt.str("");
            txt.clear();
            txt << std::fixed << std::setprecision(2) << answer.score[i] * 100 << "%";

            x = x + PANEL_PREDICTION + GAP;

            DrawTextEx(font, txt.str().c_str(), {x, y}, FONT_SIZE_SMALL, 2,WHITE);
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

        x = x_right;
        y = y + GAP;

        Color answar_color;
        if (answer.answer) {
            answar_color = GREEN;
        }
        else {
            answar_color = RED;
        }

        DrawTextEx(font, txt.str().c_str(), {x,y}, FONT_SIZE_SMALL, 2,answar_color);

        EndDrawing();
    }

    UnloadFont(font);
    CloseWindow();

    return 0;
}