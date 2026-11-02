#ifndef RESEAUDENEURONEFROMSCRATCH_SHAPE_HPP
#define RESEAUDENEURONEFROMSCRATCH_SHAPE_HPP
#include <cassert>
#include "math/Types.hpp"

class Shape {
public:
    Shape(Index c, Index h ,Index w) {
        assert((c > 0 && w > 0 && h > 0) && "Erreur : Dimension de la structure shape incorrecte");
        channel = c;
        height = h;
        width = w;
    }

    inline Index Shape::total_values() const{
        return channel*height*width;
    }

    inline Index Shape::get_column(Index c, Index h, Index w) const {
        assert((c < channel) && (h < height) && (w < width) && (c >= 0) && (h >= 0) && (w >= 0) && "Erreur : Erreur dans les paramètres");
        return c*(height*width) + h*width + w;
    }


    inline Index get_channel() const {
        return channel;
    }

    inline Index get_height() const {
        return height;
    }

    inline Index get_width() const {
        return width;
    }
private:
    Index channel;
    Index height;
    Index width;
};

#endif //RESEAUDENEURONEFROMSCRATCH_SHAPE_HPP
