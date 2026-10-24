#include "nn/Shape.hpp"

Index Shape::total_values() const{
    return channel*height*width;
}

Index Shape::get_column(Index c, Index h, Index w) const {
    assert((c < channel) && (h < height) && (w < width) && (c >= 0) && (h >= 0) && (w >= 0) && "Erreur : Erreur dans les paramètres");
    return c*(height*width) + h*width + w;
}

Index Shape::get_channel() const {
    return channel;
}

Index Shape::get_height() const {
    return height;
}

Index Shape::get_width() const {
    return width;
}