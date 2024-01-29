#pragma once
#include "vulkanexamplebase.h"
#include "marchingCube.h"

template <typename T>
#define PLANET_DIMENSION 8
#define CHUNK_DIMENSION 16
#define WORLD_LIMIT (-PLANET_DIMENSION * CHUNK_DIMENSION) + 1
#define CHUNK_RAIDUS (CHUNK_DIMENSION >> 1) * 1.414f // box's longest diagonal / 2
#define CHUNK_COUNT (PLANET_DIMENSION * PLANET_DIMENSION * PLANET_DIMENSION)
//#define MAX_TRI_COUNT_IN_A_CELL 4
//#define MAX_VERTEX_COUNT_IN_A_CELL MAX_TRI_COUNT_IN_A_CELL*3

bool contains(const std::vector<T>& vec, T data) {
    auto it = std::find(vec.begin(), vec.end(), data);
    return it != vec.end();
}
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 tangent;
};
struct Vertices {
    int count;
    VkBuffer buffer;
    VkDeviceMemory memory;
};
struct Chunk {
    //uint8_t flags; 
    uint8_t voxel[CHUNK_DIMENSION * CHUNK_DIMENSION * CHUNK_DIMENSION]; // 2x2x2 padding : 4x4x4 
    std::vector<MarchingCube::Cell> grid_of_cells_per_chunk;
    std::vector<MarchingCube::TRIANGLE> tri_list_per_chunk;
    std::vector<Vertex> vertexBuffer_per_chunk;
    struct Vertices vertices_per_chunk;
};
namespace genVoxel
{
    typedef glm::vec3 Point;
    typedef glm::vec3 Direction;
    void Sphere(Point center, float radius, std::vector<Point>& voxel);
    int pos_to_chunkIndex(glm::vec3 pos);
    int pos_to_voxelIndex(glm::vec3 pos);

    bool RayCast(glm::ivec3 start, Point rd, Chunk** chunk, std::vector<glm::vec3>& particle_pos, glm::vec3* rayHitLocation) {
        // DDA algorithm https://www.youtube.com/watch?v=NbSee-XM7WA
        // It can only handle whole numbers
        float step = 0;
        float max_step = 200.0f;
        glm::vec3 initial_ro = start;
        glm::vec3 curr_ro = initial_ro;
        unsigned int step_count = 0;
        while (step < max_step) {
            // Where I would end up if I take a step
            glm::vec3 stepX = curr_ro + rd / abs(rd.x);
            glm::vec3 stepY = curr_ro + rd / abs(rd.y);
            glm::vec3 stepZ = curr_ro + rd / abs(rd.z);
            // Walk along the axis of the step that is shortest in distance from the initial ray origin
            float distX = glm::distance(initial_ro, stepX);
            float distY = glm::distance(initial_ro, stepY);
            float distZ = glm::distance(initial_ro, stepZ);
            if (distX > distY) {
                step = distY;
            }
            else {
                step = distX;
            }
            if (step > distZ) {
                step = distZ;
            }
            glm::vec3 rayLocation = initial_ro + step * rd;
            int chunk_index = pos_to_chunkIndex(rayLocation);
            if (chunk_index < 0) { // out of bound
                return false;
            }
            // place a particle in the ray's path
            //if (step_count < particle_pos.size() / 2) {
            //    particle_pos[step_count] = rayLocation;
            //}
            //step_count++;

            Chunk* target_chunk = chunk[chunk_index];
            int voxelIndex = pos_to_voxelIndex(rayLocation);
            if (target_chunk->voxel[voxelIndex] & 1) {
                // found a present voxel
                *rayHitLocation = rayLocation;
                return true;
            }
            else {
                curr_ro = rayLocation;
            }
        }
        return false;
    }
    int return_index(glm::vec3 vec) {
        return ((int)vec.z * CHUNK_DIMENSION * CHUNK_DIMENSION) + ((int)vec.y * CHUNK_DIMENSION) + (int)vec.x;
    }
    int return_index(int x, int y, int z) {
        return (z * CHUNK_DIMENSION * CHUNK_DIMENSION) + (y * CHUNK_DIMENSION) + x;
    }
    int pos_to_chunkIndex(glm::vec3 pos) {
        // Boundnary check
        if (pos.x > 0 || pos.y > 0 || pos.z > 0 || pos.x < WORLD_LIMIT || pos.y < WORLD_LIMIT || pos.z < WORLD_LIMIT) {
            return -1;
        }
        int x_chunk_coord = -((int)pos.x / (CHUNK_DIMENSION));
        int y_chunk_coord = -((int)pos.y / (CHUNK_DIMENSION));
        int z_chunk_coord = -((int)pos.z / (CHUNK_DIMENSION));
        int chunkIndex = ((z_chunk_coord * PLANET_DIMENSION * PLANET_DIMENSION) + (y_chunk_coord * PLANET_DIMENSION) + x_chunk_coord);
        // Boudnary check
        //if (0 <= chunkIndex && chunkIndex < CHUNK_COUNT) {
        //    return chunkIndex;
        //}
        //else {
        //    return -1;
        //}
        return chunkIndex;
    }
    glm::vec3 chunkIndex_to_pos(int index) {
        int z = index / (PLANET_DIMENSION * PLANET_DIMENSION);
        index -= (z * PLANET_DIMENSION * PLANET_DIMENSION);
        int y = index / PLANET_DIMENSION;
        int x = index % PLANET_DIMENSION;
        return glm::vec3(x, y, z);
    }

    int pos_to_voxelIndex(glm::vec3 pos) {
        //int morton_code = 0;
        int x_voxel_coord = -((int)pos.x % (CHUNK_DIMENSION));
        int y_voxel_coord = -((int)pos.y % (CHUNK_DIMENSION));
        int z_voxel_coord = -((int)pos.z % (CHUNK_DIMENSION));

        //return return_index(x_voxel_coord, y_voxel_coord, z_voxel_coord);
        return ((z_voxel_coord * CHUNK_DIMENSION * CHUNK_DIMENSION) + (y_voxel_coord * CHUNK_DIMENSION) + x_voxel_coord);
    }
    glm::vec3 voxelIndex_to_pos(int index) {
        int z = index / (CHUNK_DIMENSION * CHUNK_DIMENSION);
        index -= (z * CHUNK_DIMENSION * CHUNK_DIMENSION);
        int y = index / CHUNK_DIMENSION;
        int x = index % CHUNK_DIMENSION;

        return glm::vec3(x,y,z);
    }
    
    void Remove_Voxel(Point target, Chunk** chunk, std::unordered_set<int>& damagedChunkIndices) {
        std::vector<Point> voxel_pos_to_be_removed;
        Sphere(target, 5.0f, voxel_pos_to_be_removed);
        for (int i = 0; i < voxel_pos_to_be_removed.size(); i++) {
            glm::vec3 voxelPos = voxel_pos_to_be_removed[i];
            int chunk_index = pos_to_chunkIndex(voxelPos);
            if (chunk_index < 0) { // out of bound
            }
            else {
                damagedChunkIndices.insert(chunk_index);
                Chunk* target_chunk = chunk[chunk_index];
                int voxelIndex = pos_to_voxelIndex(voxelPos);
                target_chunk->voxel[voxelIndex] = 0;
            }
        }
    }
    void Fill_Chunk(Chunk* chunk)
    {
        // Assuming we initialized all elements to 0, we can ignore the padding
        for (int x = 1; x < CHUNK_DIMENSION-1; x++) {
            for (int y = 1; y < CHUNK_DIMENSION-1; y++) {
                for (int z = 1; z < CHUNK_DIMENSION-1; z++) {
        //for (int x = 0; x < CHUNK_DIMENSION - 0; x++) {
        //    for (int y = 0; y < CHUNK_DIMENSION - 0; y++) {
        //        for (int z = 0; z < CHUNK_DIMENSION - 0; z++) {
                    //uint8_t byte = 0;
                    //byte |= 1; // present bit
                    chunk->voxel[(z * CHUNK_DIMENSION * CHUNK_DIMENSION) + (y * CHUNK_DIMENSION) + x] = 1;
                }
            }
        }
    }
    void Fill_VoxelBuffer_based_on_Chunk(Chunk* chunk, uint8_t index, std::vector<Point>& voxel) {
        for (int x = 1; x < CHUNK_DIMENSION - 1; x++) {
            for (int y = 1; y < CHUNK_DIMENSION - 1; y++) {
                for (int z = 1; z < CHUNK_DIMENSION - 1; z++) {
                    uint8_t presentBit = 1;
                    if ( chunk->voxel[(z * CHUNK_DIMENSION * CHUNK_DIMENSION) + (y * CHUNK_DIMENSION) + x] & presentBit) { // not found
                        Point voxelPosition = Point(x, y, z) + chunkIndex_to_pos(index) * (float)(CHUNK_DIMENSION);
                        Point voxelPosition_VulkanCoord = Point(-voxelPosition.y, -voxelPosition.z, -voxelPosition.x);
                        voxel.push_back( voxelPosition_VulkanCoord );
                    }
                }
            }
        }
    }

    void Cube(Point center, float radius, std::vector<Point>& voxel)
    {
        for (int x = -radius + center.x; x < radius + center.x; x++) {
            for (int y = -radius + center.y; y < radius + center.y; y++) {
                for (int z = -radius + center.z; z < radius + center.z; z++) {
                    std::vector<Point>::iterator it = std::find(voxel.begin(), voxel.end(), Point(x, y, z));
                    if (it == voxel.end()) { // not found
                        voxel.push_back(Point(x, y, z));
                    }
                }
            }
        }
    }
    
    void Sphere_Chunk(Chunk* chunk)
    {
        int RADIUS = (CHUNK_DIMENSION >> 1) - 1;
        for (int x = 1; x < CHUNK_DIMENSION - 1; x++) {
            for (int y = 1; y < CHUNK_DIMENSION - 1; y++) {
                for (int z = 1; z < CHUNK_DIMENSION - 1; z++) {
                    if ((x - RADIUS) * (x - RADIUS) + (y - RADIUS) * (y - RADIUS) + (z - RADIUS) * (z - RADIUS) < RADIUS * RADIUS) {
                        chunk->voxel[(z * CHUNK_DIMENSION * CHUNK_DIMENSION) + (y * CHUNK_DIMENSION) + x] |= 1; // present bit
                    }
                }
            }
        }
    }
    void subtractCube(Chunk* chunk, unsigned int RADIUS)
    {
        for (int x = 1; x < CHUNK_DIMENSION - 1; x++) {
            for (int y = 1; y < CHUNK_DIMENSION - 1; y++) {
                for (int z = 1; z < CHUNK_DIMENSION - 1; z++) {
                    if ((x - RADIUS) * (x - RADIUS) + (y - RADIUS) * (y - RADIUS) + (z - RADIUS) * (z - RADIUS) < RADIUS * RADIUS) {
                        chunk->voxel[(z * CHUNK_DIMENSION * CHUNK_DIMENSION) + (y * CHUNK_DIMENSION) + x] &= 0xFE; // inverse present bit;
                    }
                    
                }
            }
        }
    }
    void Sphere_remove_duplicate(Point center, float radius, std::vector<Point>& voxel)
    {
        for (int x = -radius + center.x; x < radius + center.x; x++) {
            for (int y = -radius + center.y; y < radius + center.y; y++) {
                for (int z = -radius + center.z; z < radius + center.z; z++) {
                    if ( (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) + (z - center.z) * (z - center.z) < radius * radius) {
                        std::vector<Point>::iterator it = std::find(voxel.begin(), voxel.end(), Point(x, y, z));
                        if (it == voxel.end()) { // not found
                            voxel.push_back(Point(x, y, z));
                        }
                    }
                }
            }
        }
    }
    void Sphere(Point center, float radius, std::vector<Point>& voxel)
    {
        for (int x = -radius + center.x; x < radius + center.x; x++) {
            for (int y = -radius + center.y; y < radius + center.y; y++) {
                for (int z = -radius + center.z; z < radius + center.z; z++) {
                    if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) + (z - center.z) * (z - center.z) < radius * radius) {
                        voxel.push_back(Point(x, y, z));
                    }
                }
            }
        }
    }
    void subtractSphere(Point center, float radius, std::vector<Point>& voxel)
    {
        for (int x = -radius + center.x; x < radius + center.x; x++) {
            for (int y = -radius + center.y; y < radius + center.y; y++) {
                for (int z = -radius + center.z; z < radius + center.z; z++) {
                    if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) + (z - center.z) * (z - center.z) < radius * radius) {
                        std::vector<Point>::iterator it = std::find(voxel.begin(), voxel.end(), Point(x, y, z));
                        if (it != voxel.end()) { // found
                            voxel.erase(voxel.begin() + std::distance(voxel.begin(), it));
                        }
                    }
                }
            }
        }
    }
    void Line(Point center, int height, std::vector<Point>& voxel)
    {
        for (int i = 0; i < height; i++) {
            std::vector<Point>::iterator it = std::find(voxel.begin(), voxel.end(), center + glm::vec3(0.0, i, 0.0));
            if (it == voxel.end()) { // not found
                voxel.push_back( center + glm::vec3(0.0, i, 0.0) );
            }
        }
    }

    void subtract_Y_axis(Point center, int height, std::vector<Point>& voxel)
    {
        for (int i = 0; i < height; i++) {
            std::vector<Point>::iterator it = std::find(voxel.begin(), voxel.end(), center + glm::vec3(0.0, i, 0.0));
            if (it != voxel.end()) { // found
                voxel.erase(voxel.begin() + std::distance(voxel.begin(), it));
            }
        }
    }

    void subtractCube(Point center, float radius, std::vector<Point>& voxel)
    {
        for (int x = -radius + center.x; x < radius + center.x; x++) {
            for (int y = -radius + center.y; y < radius + center.y; y++) {
                for (int z = -radius + center.z; z < radius + center.z; z++) {
                    std::vector<Point>::iterator it = std::find(voxel.begin(), voxel.end(), Point(x, y, z));
                    if (it != voxel.end()) { // found
                        voxel.erase(voxel.begin() + std::distance(voxel.begin(), it));
                    }
                }
            }
        }
    }

}